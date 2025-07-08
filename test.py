import numpy as np
import os 
import geopandas as gpd
import matplotlib.pyplot as plt
import pickle as pkl
if __name__ == '__main__':
    cached_pattern_path = 'cache/pattern_Singapore_100_with_type.pkl'
    if os.path.exists(cached_pattern_path):
        with open(cached_pattern_path, 'rb') as f:
            patterns = pkl.load(f)
    
        svi_cnt = []
        poi_cnt = []
        build_cnt = []
        for p in patterns:
            svi_cnt.append(p['svi_emb'].shape[0] if  p['svi_emb'] is not None else 0)
            poi_cnt.append( p['poi_feature'].shape[0] if  p['poi_feature'] is not None else 0)
            build_cnt.append(p['building_feature'].shape[0] if  p['building_feature'] is not None else 0)
    fig = plt.figure()
    # plt.plot(svi_cnt)
    # plt.plot(poi_cnt)
    plt.plot(build_cnt)
    plt.savefig('test.png')