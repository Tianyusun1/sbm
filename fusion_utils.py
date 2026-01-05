import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 创新点 3 核心工具: 正交子空间投影与损失函数
# ==============================================================================

class OrthogonalProjector(nn.Module):
    """
    正交投影模块：将融合后的特征强制拆分到两个互斥的子空间。
    空间 A: 身份/性别 (Identity/Gender) - 静态生理特征
    空间 B: 情绪/行为 (Emotion/Affiliative) - 动态交互特征
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 定义两个投影矩阵
        # 使用 Xavier 初始化保证分布均匀
        self.proj_identity = nn.Linear(input_dim, output_dim, bias=False)
        self.proj_behavior = nn.Linear(input_dim, output_dim, bias=False)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.proj_identity.weight)
        nn.init.orthogonal_(self.proj_behavior.weight)

    def forward(self, x):
        # x shape: [Batch, Dim]
        feat_id = self.proj_identity(x)
        feat_emo = self.proj_behavior(x)
        return feat_id, feat_emo

def compute_orthogonal_loss(projector_module: OrthogonalProjector, device='cuda'):
    """
    计算正交性惩罚损失 (Orthogonal Regularization Loss)
    目标: 最小化两个投影矩阵权重的重叠 (Frobenius Norm of Product)
    Loss = || W_id^T @ W_behavior ||_F
    """
    w_id = projector_module.proj_identity.weight      # [Out, In]
    w_beh = projector_module.proj_behavior.weight     # [Out, In]
    
    # 计算 W_id * W_beh^T (注意维度的转置关系，取决于你想约束行空间还是列空间)
    # 这里我们约束列空间正交，即不同的特征维度负责不同的任务
    # w_id.T: [In, Out], w_beh: [Out, In] -> 结果 [In, In]
    # 我们希望两个矩阵关注输入特征的不同部分
    
    product = torch.mm(w_id, w_beh.t())
    
    # Frobenius 范数
    loss = torch.norm(product, p='fro')
    
    return loss

# ==============================================================================
# 创新点 1 核心工具: 音频幻觉门控
# ==============================================================================

class AudioHallucinationGate(nn.Module):
    """
    用于 Inverse-KAN 的门控机制。
    根据音频的全局特征，决定“幻觉残差”注入视觉特征的强度。
    """
    def __init__(self, audio_dim, gate_ratio=0.25):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(audio_dim, int(audio_dim * gate_ratio)),
            nn.ReLU(),
            nn.Linear(int(audio_dim * gate_ratio), 1),
            nn.Sigmoid() # 输出 0~1 的权重系数
        )

    def forward(self, audio_feat, visual_feat, ghost_residual):
        """
        Args:
            audio_feat: 音频特征 [B, L_aud, D] -> 用于计算门控权重
            visual_feat: 原始视觉特征 [B, L_vis, D]
            ghost_residual: Inverse-KAN 生成的幻觉残差 [B, L_vis, D]
        Returns:
            V_corrected: 注入残差后的视觉特征
        """
        # 对音频特征取均值作为 Context
        ctx = audio_feat.mean(dim=1) # [B, D]
        
        # 计算注入系数 alpha
        alpha = self.gate_net(ctx).unsqueeze(1) # [B, 1, 1]
        
        # 残差注入公式: V' = V + alpha * V_ghost
        v_corrected = visual_feat + alpha * ghost_residual
        
        return v_corrected

# ==============================================================================
# 创新点 2 核心工具: 序列对齐辅助
# ==============================================================================

def align_audio_energy_to_visual(audio_energy, target_len):
    """
    将提取的音频能量序列 (Audio Energy) 对齐到视觉序列的长度。
    用于 AG-Mamba 的步长控制。
    
    Args:
        audio_energy: [Batch, Audio_Len] or [Batch, Audio_Len, 1]
        target_len: 目标长度 (Combined Sequence Length or Visual Length)
    Returns:
        aligned_energy: [Batch, Target_Len, 1]
    """
    if audio_energy.dim() == 2:
        audio_energy = audio_energy.unsqueeze(1) # [B, 1, L] for interpolate
        
    # interpolate 需要 [B, C, L] 格式
    # 假设输入是 [B, L, 1]，先转置
    if audio_energy.shape[-1] == 1:
        audio_energy = audio_energy.permute(0, 2, 1) # [B, 1, L_aud]

    # 线性插值对齐
    aligned = F.interpolate(
        audio_energy, 
        size=target_len, 
        mode='linear', 
        align_corners=False
    ) # [B, 1, Target_Len]

    # 转回 [B, Target_Len, 1]
    return aligned.permute(0, 2, 1)