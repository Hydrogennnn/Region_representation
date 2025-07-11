"""
This file is to generate embedding for SVI.
"""
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from datasets import load_dataset
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from PIL import Image
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset


from svi_utils import *

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class SVIEncoder():
    def __init__(self, gpu=-1):
        self.device = torch.device(f'cuda:{gpu}' if gpu >= 0 else 'cpu')
        self.model_name = "./clip-vit-large-patch14"
        # self.model_name = "openai/clip-vit-base-patch32"
        self.model = CLIPVisionModelWithProjection.from_pretrained(
            self.model_name, device_map=self.device)
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, device=self.device)

    def preprocess_dataset(self, dataset):
        """
        dataset: Huggingface dataset object.
        """
        dataset = dataset.map(lambda x: self.processor(images=x["image"], return_tensors="pt"),
                              batched=True)
        dataset.set_format(type='torch', columns=['pixel_values'])
        return dataset

    def embed_images(self, dataset):
        """
        dataset: Huggingface dataset object.
        """

        inference_dataloader = DataLoader(
            dataset, batch_size=64, shuffle=False, num_workers=1)

        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for batch in inference_dataloader:
                for k in batch.keys():
                    batch[k] = batch[k].to(self.device)
                outputs = self.model(**batch)
                image_embeds = outputs.image_embeds
                embeddings.append(image_embeds)
        embeddings = torch.cat(embeddings, dim=0)
        embeddings = embeddings.to('cpu')
        return embeddings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='Singapore') # Singapore / NYC
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--svi-dir', type=str, default='SVI')
    parser.add_argument('--svi-pos', type=str, default='svi_sampling_point.csv')
    parser.add_argument('--sav-dir', type=str, default='svi_emb')
    parser.add_argument('--distance', type=float, default=20.0)
    args = parser.parse_args()

    data_dir = f"../data/processed/{args.city}"
    #../data/processed/Singapore
    svi_dir = os.path.join(data_dir, args.svi_dir)
    svi_pos_dir = os.path.join(svi_dir, args.svi_pos)
    save_dir = os.path.join(data_dir, args.sav_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)


    # Dump `objectid` and `angle`
    metadata_path = os.path.join(save_dir, 'im_metadata.csv')
        

    #Generate Embedding & metadata
    embedding_path = os.path.join(save_dir, 'embedding.pt')
    if not os.path.exists(embedding_path) or not os.path.exists(metadata_path):
        # Load hugging face dataset
        print("Loading CLIP...")
        generate_imagefolder_metadata(svi_dir)
        dataset = load_dataset("imagefolder", data_dir=svi_dir, split='train')
        print(dataset)
        print("Encoding....")
        svi_enc = SVIEncoder(gpu=args.gpu)
        processed_dataset = svi_enc.preprocess_dataset(dataset)
        embeddings = svi_enc.embed_images(processed_dataset)
        torch.save(embeddings, embedding_path)
        print("Embeddings have been saved!")
        # generate metadata
        metadata_df = {'objectid': dataset['objectid'], 'angle': dataset['angle']}
        metadata_df = pd.DataFrame(metadata_df)
        metadata_df.to_csv(metadata_path, index=False)
        print("metadata have been saved!")
    else:
        embeddings = torch.load(embedding_path)
        metadata_df = pd.read_csv(metadata_path)
        print("Embeddings & metadata have been loaded!")


if __name__=='__main__':
    # 读取CSV文件
    # df = pd.read_csv('data/processed/Singapore/SVI/svi_sampling_point.csv')  # 替换为你的文件路径
    #
    # # 提取X和Y坐标
    # x = df['POINT_X']
    # y = df['POINT_Y']
    #
    # # 绘制散点图
    # # plt.figure(figsize=(8, 6))
    # # plt.scatter(x, y, c='blue', s=1)  # c是颜色，s是点的大小
    # # plt.xlabel('POINT_X')
    # # plt.ylabel('POINT_Y')
    # # plt.title('Scatter Plot of POINT_X and POINT_Y')
    # # plt.grid(True)
    # # plt.axis('equal')  # 保持XY轴刻度一致
    # # plt.show()
    #
    # gdf = gpd.GeoDataFrame(
    #     df, geometry=gpd.points_from_xy(df['POINT_X'], df['POINT_Y']))
    #
    # world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    #
    # ax = world.plot(
    #     color='white', edgecolor='black', figsize=(30, 30))
    #
    # gdf.plot(ax=ax, color='red')
    # # 设置显示范围为新加坡区域
    # plt.xlim(103.6, 104.1)
    # plt.ylim(1.2, 1.5)
    #
    # plt.show()

    # # 读取CSV中的采样点
    # df = pd.read_csv('data/processed/Singapore/SVI/svi_sampling_point.csv')
    # gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['POINT_X'], df['POINT_Y']), crs='EPSG:4326')

    # # 读取新加坡城市边界（Shapefile）
    # boundary = gpd.read_file('data/projected/Singapore/boundary/Singapore.shp')

    # # 如果边界数据不是WGS84（EPSG:4326），需投影转换（常见为EPSG:3414）
    # if boundary.crs != "EPSG:4326":
    #     boundary = boundary.to_crs("EPSG:4326")

    # # 绘图
    # fig, ax = plt.subplots(figsize=(10, 10))
    # boundary.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=1)
    # gdf.plot(ax=ax, color='red', markersize=0.1)

    # # 设置坐标轴范围（聚焦新加坡）
    # plt.xlim(103.6, 104.1)
    # plt.ylim(1.2, 1.5)

    # plt.title('SVI Sampling Points in Singapore with City Boundary')
    # plt.xlabel('Longitude')
    # plt.ylabel('Latitude')
    # plt.grid(True)
    # plt.axis('equal')
    # plt.show()



    main()
