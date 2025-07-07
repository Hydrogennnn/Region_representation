"""
@file: main.py
@time: 2022/09/21
"""
import argparse

import numpy as np
import torch

from data_util.dataset import CityData
from model.regiondcl import PatternEncoder, RegionEncoder
from model.trainer import PatternTrainer, RegionTrainer
import swanlab
import os
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='Singapore', help='City name, can be Singapore or NYC')
    parser.add_argument('--no_random', action='store_true', help='Whether to disable random points')
    parser.add_argument('--fixed', action='store_true', help='Whether to disable adaptive margin')
    parser.add_argument('--dim', type=int, default=64, help='Dimension of output representation')
    parser.add_argument('--d_feedforward', type=int, default=1024)
    parser.add_argument('--building_head', type=int, default=8)
    parser.add_argument('--building_layers', type=int, default=2)
    parser.add_argument('--building_dropout', type=float, default=0.2)
    parser.add_argument('--building_activation', type=str, default='relu')
    parser.add_argument('--bottleneck_head', type=int, default=8)
    parser.add_argument('--bottleneck_layers', type=int, default=2)
    parser.add_argument('--bottleneck_dropout', type=float, default=0.2)
    parser.add_argument('--bottleneck_activation', type=str, default='relu')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--save_name', type=str, default='')
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--use_svi', action='store_true', default=False)
    parser.add_argument('--svi_drop', type=float, default=0.1)
    parser.add_argument('--lamb', type=int, default=100)
    parser.add_argument('--first_epoch', type=int, default=20)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--radius', type=int, default=100)
    return parser.parse_args()

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    args = parse_args()
    city_data = CityData(args.city, with_random=not args.no_random, random_radius=args.radius)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Use:', device)
    args.save_name += ("svi_" if args.use_svi else "") + \
        f"dim{args.dim}-lambda{args.lamb}-lr{args.lr}-svi_drop{args.svi_drop}-bndrop{args.bottleneck_dropout}-r{args.radius}-seed{args.seed}"

    use_wandb = args.use_wandb
    use_svi = args.use_svi
    first_epoch = args.first_epoch
    setup_seed(args.seed) # reproduct

    pattern_encoder = PatternEncoder(d_building=city_data.building_feature_dim,
                                     d_poi=city_data.poi_feature_dim,
                                     d_svi=city_data.svi_emb_dim,
                                     d_hidden=args.dim,
                                     d_feedforward=args.d_feedforward,
                                     building_head=args.building_head,
                                     building_layers=args.building_layers,
                                     building_dropout=args.building_dropout,
                                     building_distance_penalty=1,
                                     building_activation=args.building_activation,
                                     bottleneck_head=args.bottleneck_head,
                                     bottleneck_layers=args.bottleneck_layers,
                                     bottleneck_dropout=args.bottleneck_dropout,
                                     bottleneck_activation=args.bottleneck_activation,
                                     use_svi=use_svi,
                                     svi_drop=args.svi_drop).to(device)
    
    if use_wandb:
        swanlab.init(
            project = 'Region',
            name = args.save_name,
            config=vars(args)
        )
    # Encode building pattern
    pattern_optimizer = torch.optim.Adam(pattern_encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pattern_scheduler = torch.optim.lr_scheduler.StepLR(pattern_optimizer, step_size=1, gamma=args.gamma)
    pattern_trainer = PatternTrainer(city_data, pattern_encoder, pattern_optimizer, pattern_scheduler, use_svi=use_svi,device=device)
    pattern_save_name = args.save_name + '_' + 'pattern_embedding'

    pattern_trainer.train_pattern_contrastive(epochs=first_epoch, save_name=pattern_save_name, use_wandb=use_wandb)
    region_aggregator = RegionEncoder(d_hidden=args.dim, d_head=8).to(device)
    
    region_optimizer = torch.optim.Adam(region_aggregator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # region_scheduler = torch.optim.lr_scheduler.StepLR(region_optimizer, step_size=1, gamma=args.gamma)
    region_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(region_optimizer, T_max=10,eta_min=args.lr/10)
    region_trainer = RegionTrainer(city_data, pattern_encoder, pattern_optimizer, pattern_scheduler, region_aggregator,
                                   region_optimizer, region_scheduler,device=device)
    # embeddings = pattern_trainer.get_embeddings()
    # Alternatively, you can load the trained pattern embedding
    embeddings = np.load(f'embeddings/{args.city}/{pattern_save_name}_{first_epoch}.npy')  # load pattern_embedding of stage1
    region_save_name = args.save_name + '_'+ 'RegionDCL'+f'{first_epoch}_'
    region_trainer.train_region_triplet_freeze(epochs=100, embeddings=embeddings, adaptive=not args.fixed, save_name=region_save_name,
                                               window_sizes=[1000, 2000, 3000], use_wandb=use_wandb, _lambda=args.lamb, first_epoch=first_epoch
                                               )
    print('Training finished. Embeddings have been saved in embeddings/ directory.')
