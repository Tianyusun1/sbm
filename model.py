import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union, List, Optional
from transformers import ViTForImageClassification

# ==============================================================================
# 1. 核心组件：Faster KAN & Inverse-KAN
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
        spline_basis = self.rbf(self.layernorm(x))
        B_size = x.shape[0]
        # 兼容 [B, L, D] 或 [B, D] 输入
        if x.dim() == 3:
            L_size = x.shape[1]
            spline_basis = spline_basis.view(B_size, L_size, -1)
        else:
            spline_basis = spline_basis.view(B_size, -1)
        return self.spline_linear(spline_basis)

class InverseKAN(nn.Module):
    """
    创新点 1: 音频幻觉补偿模块 [cite: 4]
    将音频 Deep Feature 映射为视觉空间的幻觉残差 ΔVghost
    """
    def __init__(self, audio_dim, visual_dim):
        super().__init__()
        self.mapping = FasterKANLayer(audio_dim, visual_dim)
        self.gate = nn.Sequential(
            nn.Linear(audio_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, aud_feat, vis_seq):
        # aud_feat: [B, 32, D] -> 取均值作为全局引导
        aud_global = aud_feat.mean(dim=1)
        ghost_residual = self.mapping(aud_global).unsqueeze(1) # [B, 1, D]
        weight = self.gate(aud_global).unsqueeze(1) # [B, 1, 1]
        
        # 将残差注入视觉序列 (V_corrected = V_origin + Gate * ΔVghost) [cite: 7]
        return vis_seq + weight * ghost_residual

# ==============================================================================
# 2. 创新组件：AG-Mamba (声化引导扫描)
# ==============================================================================

class AcousticGuidedMambaBlock(nn.Module):
    """
    创新点 2: 解决 Mamba 固定步长扫描问题 [cite: 9, 10]
    将音频能量谱作为 Mamba 的 Delta 控制信号
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.d_inner = dim * expand
        self.in_proj = nn.Linear(dim, self.d_inner * 2)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, groups=self.d_inner, 
                                kernel_size=d_conv, padding=d_conv - 1)
        
        # 修改: Delta 生成网络不再是纯静态投影，而是准备接收音频信号 [cite: 11]
        self.x_proj = nn.Linear(self.d_inner, (math.ceil(dim / 16) + d_state * 2), bias=False)
        self.dt_proj = nn.Linear(math.ceil(dim / 16), self.d_inner)
        
        # 能量引导层: 将 1D 能量映射到 Delta 空间
        self.energy_to_dt = nn.Linear(1, math.ceil(dim / 16))
        
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, dim)
        self.dt_rank = math.ceil(dim / 16)
        self.d_state = d_state

    def forward(self, x, audio_energy=None):
        B, L, D = x.shape
        x = self.norm(x)
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L].transpose(1, 2)
        x = F.silu(x)
        
        deltaBC = self.x_proj(x)
        delta, B_ssm, C_ssm = torch.split(deltaBC, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        # 注入音频能量引导 
        if audio_energy is not None:
            # audio_energy shape: [B, L, 1]
            # 能量大(急促) -> delta小(高频扫描)；能量小(平缓) -> delta大(粗略扫描)
            dt_bias = self.energy_to_dt(1.0 / (audio_energy + 1e-4)) 
            delta = delta + dt_bias
            
        delta = F.softplus(self.dt_proj(delta)) 
        
        # SSM Core Scan (与原版逻辑一致)
        A = -torch.exp(self.A_log)
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)) 
        deltaB_x = delta.unsqueeze(-1) * B_ssm.unsqueeze(2) * x.unsqueeze(-1)
        
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        ys = []
        for t in range(L):
            h = deltaA[:, t] * h + deltaB_x[:, t]
            ys.append(h)
        ys = torch.stack(ys, dim=1)
        
        y = (ys @ C_ssm.unsqueeze(2).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        y = y + self.D * x
        return self.out_proj(y * F.silu(z))

class ChimeraMambaKANBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.token_mixer = AcousticGuidedMambaBlock(dim)
        self.channel_mixer = FasterKANLayer(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, audio_energy=None):
        x = x + self.token_mixer(self.norm1(x), audio_energy=audio_energy)
        x = x + self.channel_mixer(self.norm2(x))
        return x

# ==============================================================================
# 3. 主模型：Chimera-KAN 2.0 (理毛/社交增强版)
# ==============================================================================

class AudioEncoder(nn.Module):
    def __init__(self, output_dim=768):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 8)) 
        )
        self.proj = nn.Linear(256, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = self.features(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(self.proj(x))
        # 返回序列特征 [B, 32, 768] 和 简单的能量包络估计
        energy = x.mean(dim=-1, keepdim=True) # [B, 32, 1]
        return x, energy

class MultiModalChimeraKAN(nn.Module):
    def __init__(self, vit_model,
                 emotion_num_labels: int,
                 individual_num_labels: int,
                 gender_num_labels: int,
                 unified_dim: int = 768,
                 num_chimera_layers: int = 2):
        super().__init__()
        self.vit = vit_model
        self.audio_encoder = AudioEncoder(output_dim=unified_dim)
        self.inv_kan = InverseKAN(unified_dim, unified_dim) # 创新点 1
        
        self.vis_proj = nn.Identity() 
        if vit_model.config.hidden_size != unified_dim:
            self.vis_proj = nn.Linear(vit_model.config.hidden_size, unified_dim)

        self.chimera_layers = nn.ModuleList([
            ChimeraMambaKANBlock(unified_dim) for _ in range(num_chimera_layers)
        ])
        
        # 创新点 3: 正交子空间投影层 [cite: 14, 18]
        self.proj_identity = nn.Linear(unified_dim, unified_dim) # 空间 A: 个体/性别
        self.proj_behavior = nn.Linear(unified_dim, unified_dim) # 空间 B: 情绪/行为
        
        self.emotion_classifier = FasterKANLayer(unified_dim, emotion_num_labels)
        self.individual_classifier = FasterKANLayer(unified_dim, individual_num_labels)
        self.gender_classifier = FasterKANLayer(unified_dim, gender_num_labels)
        
        self.modality_token_vis = nn.Parameter(torch.zeros(1, 1, unified_dim))
        self.modality_token_aud = nn.Parameter(torch.zeros(1, 1, unified_dim))
        nn.init.trunc_normal_(self.modality_token_vis, std=0.02)
        nn.init.trunc_normal_(self.modality_token_aud, std=0.02)

    def forward(self, pixel_values, audio_values, 
                labels=None, individual_labels=None, gender_labels=None):
        # 1. 特征提取
        vis_out = self.vit(pixel_values=pixel_values, output_hidden_states=True)
        vis_seq = self.vis_proj(vis_out.hidden_states[-1])
        aud_seq, aud_energy = self.audio_encoder(audio_values)
        
        # 2. 幻觉残差注入 (音频补偿视觉) [cite: 7, 8]
        vis_seq = self.inv_kan(aud_seq, vis_seq)
        
        # 3. 注入位置信息并拼接
        vis_seq = vis_seq + self.modality_token_vis
        aud_seq = aud_seq + self.modality_token_aud
        combined_seq = torch.cat([vis_seq, aud_seq], dim=1)
        
        # 调整能量序列长度以匹配拼接后的序列 [229]
        # 简单的插值对齐
        full_energy = F.interpolate(aud_energy.transpose(1, 2), 
                                    size=(combined_seq.shape[1]), 
                                    mode='linear').transpose(1, 2)
        
        # 4. AG-Mamba 融合扫描
        for layer in self.chimera_layers:
            combined_seq = layer(combined_seq, audio_energy=full_energy)
            
        pooled_feat = combined_seq.mean(dim=1)
        
        # 5. 正交投影解耦 [cite: 17, 18, 19]
        feat_id = self.proj_identity(pooled_feat) # 提取“长相”特征
        feat_emo = self.proj_behavior(pooled_feat) # 提取“动态互动”特征
        
        emotion_logits = self.emotion_classifier(feat_emo)
        individual_logits = self.individual_classifier(feat_id)
        gender_logits = self.gender_classifier(feat_id)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            l_emo = loss_fct(emotion_logits, labels)
            l_ind = loss_fct(individual_logits, individual_labels) if individual_labels is not None else 0
            l_gen = loss_fct(gender_logits, gender_labels) if gender_labels is not None else 0
            
            # 正交损失约束 (Force space A and B to be orthogonal)
            # Minimize ||W_id^T * W_emo||_F
            ortho_loss = torch.norm(torch.mm(self.proj_identity.weight, 
                                            self.proj_behavior.weight.t()), p='fro')
            
            loss = l_emo + l_ind + l_gen + 0.01 * ortho_loss # 0.01为正交约束权重
            
        return {
            "loss": loss,
            "emotion_logits": emotion_logits,
            "individual_logits": individual_logits,
            "gender_logits": gender_logits,
            "fused_features": pooled_feat 
        }