import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union, List, Optional
from transformers import ViTForImageClassification

# ==============================================================================
# 1. Faster KAN (Kolmogorov-Arnold Networks) - 核心组件
#    (用于替代传统的 MLP/Linear，提供更强的非线性表达能力)
# ==============================================================================

class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw):
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

class ReflectionalSwitchFunction(nn.Module):
    def __init__(self, grid_min=-2., grid_max=2., num_grids=8, denominator=0.33):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator
        self.inv_denominator = 1 / denominator

    def forward(self, x):
        diff = (x[..., None] - self.grid)
        diff_tanh = torch.tanh(diff * self.inv_denominator)
        return 1 - diff_tanh * diff_tanh

class FasterKANLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int,
                 grid_min=-2., grid_max=2., num_grids=8, denominator=0.33, **kw):
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = ReflectionalSwitchFunction(grid_min, grid_max, num_grids, denominator)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim)

    def forward(self, x):
        # KAN处理的是最后一维特征，所以它可以直接处理 [Batch, Seq_Len, Dim]
        spline_basis = self.rbf(self.layernorm(x))
        # 展平以便通过 Linear 层
        # x shape: [B, L, D] -> spline shape: [B, L, D, G] -> view [B, L, D*G]
        B_size, L_size, _ = x.shape
        spline_basis = spline_basis.view(B_size, L_size, -1)
        return self.spline_linear(spline_basis)

# ==============================================================================
# 2. Mamba 核心组件 (Global Mamba)
#    (用于处理长序列的时序依赖，无需 CUDA 编译的纯 PyTorch 实现，确保兼容性)
# ==============================================================================

class GlobalMambaBlockFromMamba(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.d_inner = dim * expand
        
        self.in_proj = nn.Linear(dim, self.d_inner * 2)
        
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            groups=self.d_inner, kernel_size=d_conv, padding=d_conv - 1
        )
        
        self.x_proj = nn.Linear(self.d_inner, (math.ceil(dim / 16) + d_state * 2), bias=False)
        self.dt_proj = nn.Linear(math.ceil(dim / 16), self.d_inner, bias=True)
        
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, dim)
        
        self.dt_rank = math.ceil(dim / 16)
        self.d_state = d_state

    def forward(self, x):
        # x: [Batch, Seq_Len, Dim]
        B, L, D = x.shape
        input_x = x
        x = self.norm(x)
        
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # [B, L, d_inner]
        
        # Conv1d 需要 [B, Channel, Seq_Len]
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)
        
        x = F.silu(x)
        
        # SSM Parameter Projection
        deltaBC = self.x_proj(x) # [B, L, dt_rank + 2*d_state]
        delta, B_ssm, C_ssm = torch.split(deltaBC, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        delta = F.softplus(self.dt_proj(delta)) # [B, L, d_inner]
        
        # SSM Core (Scan) 
        A = -torch.exp(self.A_log) # [d_inner, d_state]
        
        # 预计算 delta * A 和 delta * B * x
        # [B, L, d_inner, d_state]
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)) 
        deltaB_x = delta.unsqueeze(-1) * B_ssm.unsqueeze(2) * x.unsqueeze(-1)
        
        # Scan 过程 (Recurrence)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        ys = []
        for t in range(L):
            h = deltaA[:, t] * h + deltaB_x[:, t]
            ys.append(h)
        ys = torch.stack(ys, dim=1) # [B, L, d_inner, d_state]
        
        # y = ys @ C + D * x
        y = (ys @ C_ssm.unsqueeze(2).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        y = y + self.D * x
        
        output = y * F.silu(z)
        output = self.out_proj(output)
        
        return output 

# ==============================================================================
# 3. 骚气组合: Chimera Mamba-KAN Block
#    (方案1+方案2结合体：Mamba 做 Token 混合，KAN 做 Channel 混合)
# ==============================================================================

class ChimeraMambaKANBlock(nn.Module):
    """
    终极缝合怪 Block:
    1. Token Mixer: 使用 Global Mamba 处理长序列交互 (Visual + Audio)
    2. Channel Mixer: 使用 Faster KAN 替代 MLP 处理特征变换
    """
    def __init__(self, dim):
        super().__init__()
        # Token Mixer (Mamba)
        self.token_mixer = GlobalMambaBlockFromMamba(dim)
        
        # Channel Mixer (KAN)
        # 这里用 KAN 替代了传统的 Feed-Forward Network (FFN/MLP)
        self.channel_mixer = FasterKANLayer(dim, dim)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # 1. Mamba 混合 (序列交互) - 残差连接
        x = x + self.token_mixer(self.norm1(x))
        
        # 2. KAN 混合 (特征变换) - 残差连接
        x = x + self.channel_mixer(self.norm2(x))
        
        return x

# ==============================================================================
# 4. 辅助模块: Audio Encoder
# ==============================================================================

class AudioEncoder(nn.Module):
    def __init__(self, output_dim=768):
        super().__init__()
        # CNN 提取 Mel-Spectrogram 特征
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            # 输入 128x128 -> 最终池化到 4x8 的大小
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 8)) 
            # Output: [B, 256, 4, 8] -> Flatten -> [B, 256, 32] 序列长度为 32
        )
        self.proj = nn.Linear(256, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        # x: [B, 1, 128, 128]
        x = self.features(x) # [B, 256, 4, 8]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # [B, H*W, C] -> [B, 32, 256]
        x = self.proj(x) # [B, 32, output_dim]
        x = self.norm(x)
        return x

# ==============================================================================
# 5. 主模型: Chimera-KAN Network (嵌合体网络)
# ==============================================================================

class MultiModalChimeraKAN(nn.Module):
    def __init__(self, vit_model,
                 emotion_num_labels: int,
                 unified_dim: int = 768,
                 num_chimera_layers: int = 2):
        """
        Chimera-KAN Network:
        结合了 ViT (Vision), CNN (Audio), Mamba (Fusion), KAN (Classification) 的多模态架构。
        """
        super().__init__()
        
        # 1. 视觉 Backbone (ViT)
        self.vit = vit_model
        
        # 2. 音频 Backbone (Audio Encoder)
        self.audio_encoder = AudioEncoder(output_dim=unified_dim)
        
        # 3. 模态对齐投影 (确保维度一致，通常都是 768)
        self.vis_proj = nn.Identity() 
        if vit_model.config.hidden_size != unified_dim:
            self.vis_proj = nn.Linear(vit_model.config.hidden_size, unified_dim)

        # 4. Chimera 融合模块 (核心创新点)
        # 方案1: 序列拼接 + Mamba 处理长序列
        # 方案2: KAN 替代 MLP
        self.chimera_layers = nn.ModuleList([
            ChimeraMambaKANBlock(unified_dim) for _ in range(num_chimera_layers)
        ])
        
        # 5. 最终分类头 (使用 FasterKAN)
        self.emotion_classifier = FasterKANLayer(unified_dim, emotion_num_labels)
        
        # 6. 可学习的模态类型 Embedding (类似 BERT 的 Segment Embedding)
        # 用来告诉 Mamba 哪段是视频，哪段是音频
        self.modality_token_vis = nn.Parameter(torch.zeros(1, 1, unified_dim))
        self.modality_token_aud = nn.Parameter(torch.zeros(1, 1, unified_dim))
        nn.init.trunc_normal_(self.modality_token_vis, std=0.02)
        nn.init.trunc_normal_(self.modality_token_aud, std=0.02)

    def forward(self, pixel_values, audio_values, labels=None):
        # --- 1. 特征提取 ---
        # Visual: [B, 197, 768] (包含 CLS)
        vis_out = self.vit(pixel_values=pixel_values, output_hidden_states=True)
        vis_seq = vis_out.hidden_states[-1] 
        vis_seq = self.vis_proj(vis_seq)
        
        # Audio: [B, 32, 768]
        aud_seq = self.audio_encoder(audio_values)
        
        # --- 2. 注入模态位置信息 (让 Mamba 知道在读什么) ---
        vis_seq = vis_seq + self.modality_token_vis
        aud_seq = aud_seq + self.modality_token_aud
        
        # --- 3. 序列拼接 (Chimera Early Fusion) ---
        # 方案1核心：形成长序列，交给 Mamba 处理上下文
        # 结果: [B, 197+32, 768] = [B, 229, 768]
        combined_seq = torch.cat([vis_seq, aud_seq], dim=1)
        
        # --- 4. Chimera Mamba-KAN 处理 ---
        # Mamba 和 KAN 交替发力，深度融合
        for layer in self.chimera_layers:
            combined_seq = layer(combined_seq)
            
        # --- 5. 聚合与分类 ---
        # 对融合后的序列进行 Global Average Pooling
        pooled_feat = combined_seq.mean(dim=1) # [B, 768]
        
        # 方案2核心：KAN 分类头
        emotion_logits = self.emotion_classifier(pooled_feat)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(emotion_logits, labels)
            
        return {
            "loss": loss,
            "logits": emotion_logits,
            # 返回中间特征方便可视化
            "fused_features": pooled_feat 
        }