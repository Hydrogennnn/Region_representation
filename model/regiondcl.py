import torch
import torch.nn as nn
import torch.nn.functional as F

from model.biased_attention import BiasedMultiheadAttention
from .RegionMHA import RegionMHA

EPS = 1e-15


class BiasedEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(BiasedEncoderLayer, self).__init__()
        self.self_attn = BiasedMultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu if activation == "relu" else F.gelu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(BiasedEncoderLayer, self).__setstate__(state)

    def forward(self, src, bias, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src = self.norm1(src)
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask, bias=bias)[0]
        src = src + self.dropout1(src2)
        src = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src


class DistanceBiasedTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout, activation,
                 distance_penalty):
        super(DistanceBiasedTransformer, self).__init__()
        self.num_layers = num_layers
        self.distance_layers = nn.ModuleList([
            BiasedEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation)
            for _ in range(num_layers)])
        self.distance_penalty = nn.Parameter(torch.normal(0, distance_penalty, [nhead], requires_grad=True))

    def forward(self, x, src_key_padding_mask, distance):
        # Building_feature: [seq_len, batch_size, d_model]
        # Calculate distance bias. Each head will have a distance bias
        distance_bias = self.distance_penalty.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * -distance.unsqueeze(1)
        hidden = x
        for layer in self.distance_layers:
            hidden = layer(hidden, distance_bias, src_key_padding_mask=src_key_padding_mask)
        return hidden


class ProjectionHead(nn.Module):
    """
    The projection head is used to
    transform the space and the time embeddings to the same embedding space with the same dimensionality.
    """
    def __init__(self, embedding_dim, projection_dim=256, dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()  # apply the Gaussian Error Linear Units function
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)  # apply Layer Normalization over a mini-batch of inputs

    def forward(self, x):
        """
        :param x: tensor, shape [batch_size, embedding_dim]
        :return: tensor, shape [batch_size, projection_dim]
        """
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class PatternEncoder(nn.Module):
    def __init__(self, d_building, d_poi,d_svi, d_hidden, d_feedforward,
                 building_head, building_layers,
                 building_dropout, building_activation, building_distance_penalty,
                 bottleneck_head, bottleneck_layers, bottleneck_dropout, bottleneck_activation,
                 use_svi=False, svi_drop=0.0, ):
        super(PatternEncoder, self).__init__()
        self.building_projector = nn.Linear(d_building, d_hidden)
        self.poi_projector = nn.Linear(d_poi, d_hidden)
        self.poi_gate = nn.Linear(d_poi, 1)
        self.use_svi = use_svi
        if self.use_svi:
            self.svi_projector = ProjectionHead(d_svi, d_hidden, svi_drop)
            self.svi_gate = nn.Linear(d_svi, 1)
            # self.svi_projector = nn.Sequential(
            #     nn.Linear(d_svi, d_hidden),
            #     # nn.ReLU(),  # 手动添加激活函数
            #     # nn.LayerNorm(normalized_shape=d_hidden)
            #     # nn.Dropout(p=svi_drop)
            # )
            
        self.building_encoder = DistanceBiasedTransformer(d_model=d_hidden,
                                                          nhead=building_head,
                                                          num_layers=building_layers,
                                                          dim_feedforward=d_feedforward, dropout=building_dropout,
                                                          activation=building_activation,
                                                          distance_penalty=building_distance_penalty)
        self.bottleneck = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_hidden, nhead=bottleneck_head, dim_feedforward=d_feedforward,
                                       dropout=bottleneck_dropout, activation=bottleneck_activation), bottleneck_layers)
        # self.svi_building_cross_atten = nn.MultiheadAttention(embed_dim=d_hidden,
        #                                                       num_heads=8)
        # self.poi_building_cross_atten = nn.MultiheadAttention(embed_dim=d_hidden,
        #                                                       num_heads=8)
        self.cross_attn = RegionMHA(d_model=d_hidden, num_heads=8)

    def forward(self, building_feature, building_mask, xy, poi_feature, poi_mask, svi_emb, svi_mask):
        origin_feature = self.get_all(building_feature, building_mask, xy, poi_feature, poi_mask, svi_emb, svi_mask).mean(dim=0)  # (seq_len, batch_size, d)
        return origin_feature
        # if not self.use_svi:
        #     return origin_feature.mean(dim=0)
        # else:
        #     svi_emb = self.svi_projector(svi_emb)
        #     return torch.cat([origin_feature, svi_emb], dim=0).mean(dim=0)
    

    def get_embedding(self, building_feature, building_mask, xy, poi_feature, poi_mask, svi_emb, svi_mask):
        # add up the 0-100 dimension of building density to test the performance
        # return self.get_all(building_feature, building_mask, xy, poi_feature, poi_mask).mean(dim=0)
        origin_feature = self.get_all(building_feature, building_mask, xy, poi_feature, poi_mask, svi_emb, svi_mask).mean(dim=0)  # (seq_len, batch_size, d)
        return origin_feature
        # if not self.use_svi:
        #     return origin_feature.mean(dim=0)
        # else:
        #     svi_emb = self.svi_projector(svi_emb)
        #     return torch.cat([origin_feature, svi_emb], dim=0).mean(dim=0)

    def get_all(self, building_feature, building_mask, xy, poi_feature, poi_mask, svi_emb, svi_mask):
        building_encoding = self.building_projector(building_feature)
        batch_size = building_encoding.shape[1]
        building_loc = xy.transpose(0, 1)
        building_distance = torch.norm(building_loc.unsqueeze(2) - building_loc.unsqueeze(1), dim=3)
        # # Test the new formula
        building_distance[building_mask.unsqueeze(1) | building_mask.unsqueeze(2)] = 0
        # # get maximum_distance per pattern
        max_distance = torch.max(building_distance.view(batch_size, -1), dim=1)[0]
        normalized_distance = torch.log(
            (torch.pow(max_distance.unsqueeze(1).unsqueeze(1), 1.5) + 1) / (torch.pow(building_distance, 1.5) + 1))
        building_encoding = self.building_encoder(building_encoding, building_mask, normalized_distance)
        # ==========>
        encoding_list = [building_encoding]
        mask_list = [building_mask]
        if poi_feature is not None:
            # poi_encoding = self.poi_projector(poi_feature)
            poi_score = self.poi_gate(poi_feature)
            poi_score = F.softmax(poi_score, dim=0)
            poi_encoding = self.poi_projector(poi_feature) * poi_score

            encoding_list.append(poi_encoding)
            mask_list.append(poi_mask)
        
        if self.use_svi and svi_emb is not None:
            svi_score = self.svi_gate(svi_emb)
            svi_score = F.softmax(svi_score, dim=0)
            svi_emb = self.svi_projector(svi_emb) * svi_score #(len,b,d)

            encoding_list.append(svi_emb)
            mask_list.append(svi_mask)
        
        encoding = torch.cat(encoding_list, dim=0)
        encoding_mask = torch.cat(mask_list, dim=1)
        #============>
        # Dual Cross Attention
        # encoding_list = [building_encoding]
        # mask_list = [building_mask]

        # if poi_feature is not None and self.use_svi:
        #     poi_feature = self.poi_projector(poi_feature)
        #     svi_emb = self.svi_projector(svi_emb)
        #     # poi_svi_encoding = self.cross_attn(building_encoding, poi_feature, svi_emb, building_mask)
        #     poi_attn, svi_attn = self.cross_attn(poi_feature, svi_emb, building_encoding, building_encoding, building_mask)
        #     encoding_list.append(poi_attn)
        #     encoding_list.append(svi_attn)
        #     mask_list.append(poi_mask)
        #     mask_list.append(svi_mask)
        # else:
        #     if poi_feature is None:
        #         print("poi is None")
        #     else:
        #         print("svi is None")
        # if poi_feature is not None:
        #     poi_encoding = self.poi_projector(poi_feature)
        #     poi_encoding = self.cross_attn(poi_encoding, building_encoding, building_encoding, key_padding_mask=building_mask)
        #     encoding_list.append(poi_encoding)
        #     mask_list.append(poi_mask)
        #
        # if self.use_svi and svi_emb is not None:
        #     svi_emb = self.svi_projector(svi_emb)
        #     svi_emb = self.svi_building_cross_atten(svi_emb, building_encoding, building_encoding, key_padding_mask=building_mask)
        #     encoding_list.append(svi_emb)
        #     mask_list.append(svi_mask)


        # encoding = torch.cat(encoding_list, dim=0)
        # encoding_mask = torch.cat(mask_list, dim=1)

        # bottleneck
        bottleneck_encoding = self.bottleneck(encoding, src_key_padding_mask=encoding_mask) #第二层transformer
        # concatenate bottleneck and bottleneck embedding
        # print(bottleneck_encoding.shape)
        # exit()
        return bottleneck_encoding


class TransformerPatternEncoder(nn.Module):
    def __init__(self, d_building, d_hidden, d_feedforward,
                 building_head, building_layers,
                 building_dropout, building_activation):
        super(TransformerPatternEncoder, self).__init__()
        self.building_projector = nn.Linear(d_building, d_hidden)
        self.norm = nn.LayerNorm(d_hidden)
        self.building_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_hidden, nhead=building_head, dim_feedforward=d_feedforward,
                                       dropout=building_dropout, activation=building_activation), building_layers,
            norm=self.norm)

    def forward(self, building_feature, building_mask, xy):
        building_encoding = self.building_projector(building_feature)
        return self.building_encoder(building_encoding, src_key_padding_mask=building_mask).mean(dim=0)


class RegionEncoder(nn.Module):
    """
    In RegionEncoder, we use a simple Transformer encoder to aggregate all patterns in a region
    """

    def __init__(self, d_hidden, d_head):
        super(RegionEncoder, self).__init__()
        self.attention = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_hidden, nhead=d_head), 1)

    def forward(self, x, sigmoid=True):
        # x: [seq_len, batch_size, d_hidden]
        x = x.unsqueeze(1)
        # x = torch.cat([self.region_embedding.repeat(1, x.shape[1], 1), x], dim=0)
        x = self.attention(x).squeeze(1)
        return x

    def get_embedding(self, x):
        return self.forward(x, sigmoid=False).mean(dim=0)
