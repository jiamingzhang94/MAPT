import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AFRModule(nn.Module):
    """Attention-based Feature Refinement Module"""
    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Sigmoid()
        )
        
    def forward(self, f_i, p_i):
        # f_i: 特征 [B, N, D]
        # p_i: prompt [M, D] -> [B, M, D]
        if p_i.dim() == 2:
            p_i = p_i.unsqueeze(0).expand(f_i.shape[0], -1, -1)
            
        # 计算注意力
        q = self.q_proj(f_i)  # [B, N, D]
        k = self.k_proj(p_i)  # [B, M, D]
        v = self.v_proj(p_i)  # [B, M, D]
        
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))  # [B, N, M]
        attn = F.softmax(attn, dim=-1)
        
        # 计算refinement特征
        refined = attn @ v  # [B, N, D]
        
        # 自适应融合门控
        gate = self.gate(f_i)
        output = gate * f_i + (1 - gate) * refined
        
        return output 