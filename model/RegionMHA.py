import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#
class RegionMHA(nn.Module):
    def __init__(self, d_model, num_heads):
        super(RegionMHA, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.wq_poi = nn.Linear(d_model, d_model)
        self.wq_svi = nn.Linear(d_model, d_model)

        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout()
        self.scale = np.sqrt(self.d_k)

        self.fc = nn.Linear(d_model, d_model)

    def forward(self, Qpoi, Qsvi, K, V, key_padding_mask):
        #(seq_len, batch_size, d)
        N = Qpoi.shape[1]
        Qpoi = self.wq_poi(Qpoi).transpose(0, 1).clone().view(N, -1, self.num_heads, self.d_k).transpose(1, 2)
        Qsvi = self.wq_svi(Qsvi).transpose(0, 1).clone().view(N, -1, self.num_heads, self.d_k).transpose(1, 2)

        K = self.wk(K).transpose(0, 1).clone().view(N, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.wv(V).transpose(0, 1).clone().view(N, -1, self.num_heads, self.d_k).transpose(1, 2)

        poi_score = torch.matmul(Qpoi, K.transpose(-1, -2)) / self.scale
        svi_score = torch.matmul(Qsvi, K.transpose(-1, -2)) / self.scale #(N, num_heads, svi_len, k_len)

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            poi_score = poi_score.masked_fill_(mask, float('-inf'))
            svi_score = svi_score.masked_fill_(mask, float('-inf'))

        poi_score = torch.softmax(poi_score, dim=-1)
        svi_score = torch.softmax(svi_score, dim=-1)

        poi_score = self.dropout(poi_score)
        svi_score = self.dropout(svi_score)

        poi_out = torch.matmul(poi_score, V)#(N, num_heads, svi_len, d_v)
        svi_out = torch.matmul(svi_score, V)

        poi_out = poi_out.transpose(1, 2).contiguous().reshape(N, -1, self.d_v * self.num_heads)
        svi_out = svi_out.transpose(1, 2).contiguous().reshape(N, -1, self.d_v * self.num_heads)

        poi_out = self.fc(poi_out).transpose(0, 1).clone()
        svi_out = self.fc(svi_out).transpose(0, 1).clone()

        return poi_out, svi_out



#
# class TwoWayCrossAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#
#         # 不共享的 Query projection
#         self.q_proj_B = nn.Linear(embed_dim, embed_dim)
#         self.q_proj_C = nn.Linear(embed_dim, embed_dim)
#
#         # 共享的 Key/Value projection
#         self.k_proj = nn.Linear(embed_dim, embed_dim)
#         self.v_proj = nn.Linear(embed_dim, embed_dim)
#
#         # Output projection（可选共享）
#         self.out_proj = nn.Linear(embed_dim, embed_dim)
#
#     def forward(self, A, B, C, padding_mask):
#         # B: (seq_len_B, batch, dim)
#         # C: (seq_len_C, batch, dim)
#         # A: (seq_len_A, batch, dim)
#
#         Q_B = self.q_proj_B(B)
#         Q_C = self.q_proj_C(C)
#         K = self.k_proj(A)
#         V = self.v_proj(A)
#
#         # Cross-Attention for B
#         out_B, _ = F.multi_head_attention_forward(
#             query=Q_B, key=K, value=V,
#             embed_dim_to_check=self.embed_dim,
#             num_heads=self.num_heads,
#             in_proj_weight=None,  # 已经自己投影过
#             in_proj_bias=None,
#             bias_k=None, bias_v=None,
#             add_zero_attn=False,
#             dropout_p=0.2,
#             training=self.training,
#             need_weights=False,
#             key_padding_mask=padding_mask
#         )
#
#         # Cross-Attention for C
#         out_C, _ = F.multi_head_attention_forward(
#             query=Q_C, key=K, value=V,
#             embed_dim_to_check=self.embed_dim,
#             num_heads=self.num_heads,
#             in_proj_weight=None,
#             in_proj_bias=None,
#             bias_k=None, bias_v=None,
#             add_zero_attn=False,
#             dropout_p=0.2,
#             training=self.training,
#             need_weights=False,
#             key_padding_mask=padding_mask
#         )
#         output = torch.cat([out_B, out_C], dim=0)
#
#         # 拼接后可以做自注意力等操作
#         return output



