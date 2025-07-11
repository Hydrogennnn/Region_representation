import numpy as np
import os 
import geopandas as gpd
import matplotlib.pyplot as plt
import pickle as pkl
import torch.nn.functional as F
import torch
import torch.nn as nn
if __name__ == '__main__':
    # cached_pattern_path = 'cache/pattern_Singapore_100_with_type.pkl'
    # if os.path.exists(cached_pattern_path):
    #     with open(cached_pattern_path, 'rb') as f:
    #         patterns = pkl.load(f)
    #
    #     svi_cnt = []
    #     poi_cnt = []
    #     build_cnt = []
    #     for p in patterns:
    #         svi_cnt.append(p['svi_emb'].shape[0] if  p['svi_emb'] is not None else 0)
    #         poi_cnt.append( p['poi_feature'].shape[0] if  p['poi_feature'] is not None else 0)
    #         build_cnt.append(p['building_feature'].shape[0] if  p['building_feature'] is not None else 0)
    # fig = plt.figure()
    # # plt.plot(svi_cnt)
    # # plt.plot(poi_cnt)
    # plt.plot(build_cnt)
    # plt.savefig('test.png')
    # proj = nn.Linear(4,1)
    # x = torch.ones([3,4])
    # print(proj(x).shape)

    # x = torch.tensor([
    #     [2.0, 3.0],
    #     [1.0, 4.0],
    #     [0.5, 0.1],
    #     [-1.0, 2.0],
    #     [2.2, 0.5]
    # ])

    # 正确归一化
    # out1 = F.softmax(x, dim=0)
    #
    # # 错误理解：单元素 softmax → 全是1
    # out2 = F.softmax(x.unsqueeze(-1), dim=0)
    #
    # print(out1.shape)  # torch.Size([5, 2])
    # print(out2.shape)  # torch.Size([5, 2, 1])
    # print(out2.squeeze(-1))  # 全是1
    x=torch.tensor([[1, 2]])
    y=x.clone()
    print(torch.stack([x,y]).shape, torch.concat([x,y]).shape)
