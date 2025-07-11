"""
@file: trainer.py
@time: 2022/09/15
"""
import os
import pickle as pkl

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import random
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import swanlab
from data_util.dataset import UnsupervisedPatternDataset, FreezePatternForwardDataset, FreezePatternPretrainDataset
from model.adaptive_triplet import adaptive_triplet_loss, triplet_loss
from experiment.evaluator import land_use_inference

'''
    Trainer for RegionDCL
    Though the dual contrastive learning can be trained together,
    We found it unnecessary to do so as it is time-wasting and the performance is not significantly improved.
    Therefore, we remove the co-training and let them train together.
'''


class PatternTrainer(object):
    def __init__(self, city_data, model, optimizer, scheduler, use_svi=False, device=torch.device('cuda')):
        self.city_data = city_data
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = 10
        self.use_svi = use_svi

    def forward(self, building_feature, building_mask, xy, poi_feature, poi_mask, svi_emb, svi_mask):
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()
        building_feature = torch.from_numpy(building_feature).to(self.device)
        building_mask = torch.from_numpy(building_mask).to(self.device)
        xy = torch.from_numpy(xy).to(self.device)
        svi_emb = svi_emb.to(self.device)
        svi_mask = torch.from_numpy(svi_mask).to(self.device)
        if poi_feature is not None:
            poi_feature = torch.from_numpy(poi_feature).to(self.device)
            poi_mask = torch.from_numpy(poi_mask).to(self.device)
        return self.model(building_feature, building_mask, xy, poi_feature, poi_mask, svi_emb, svi_mask)

    def infonce_loss(self, y_pred, lamda=0.05):
        N = y_pred.shape[0]
        idxs = torch.arange(0, N, device=self.device)
        y_true = idxs + 1 - idxs % 2 * 2
        # similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
        y_pred = F.softmax(y_pred, dim=1)
        off_diag = np.ones((N, N))
        indices = np.where(off_diag)
        rec_idx = torch.LongTensor(indices[0]).to(self.device)
        send_idx = torch.LongTensor(indices[1]).to(self.device)
        senders = y_pred[send_idx]
        receivers = y_pred[rec_idx]

        similarities = 1 - F.kl_div(senders.log(), receivers, reduction='none').sum(dim=1).view(N, N)
        similarities = similarities - torch.eye(y_pred.shape[0], device=self.device) * 1e12
        similarities = similarities / lamda
        loss = F.cross_entropy(similarities, y_true)
        return torch.mean(loss)

    def train_pattern_contrastive(self, epochs, save_name, use_wandb):
        dataset = UnsupervisedPatternDataset(self.city_data)
        save_path = 'embeddings/{}/'.format(self.city_data.city)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True,
                                                   collate_fn=UnsupervisedPatternDataset.collate_fn_dropout)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False,
                                                  collate_fn=UnsupervisedPatternDataset.collate_fn)
        criterion = self.infonce_loss
        for epoch in range(1, epochs+1):
            print('epoch:',epoch, 'model_state', self.model.training)
            tqdm_batch = tqdm(train_loader, desc='Epoch {}'.format(epoch))
            losses = []
            for data in tqdm_batch:
                self.optimizer.zero_grad()
                building_feature, building_mask, xy, poi_feature, poi_mask, svi_emb, svi_mask = data
                pred = self.forward(building_feature, building_mask, xy, poi_feature, poi_mask, svi_emb, svi_mask) # embedding of each pattern
                loss = criterion(pred)
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()
                tqdm_batch.set_postfix(loss=loss.item())
                losses.append(loss.item())
            print('Epoch {}: InfoNCE Loss {}'.format(epoch, np.mean(losses)))
            if use_wandb:
                swanlab.log({
                    'pattern-loss': np.mean(losses)
                },step=epoch)
            self.scheduler.step()
            if epoch % 10 == 0:
                self.save_embedding(save_path + save_name + '_' + str(epoch) + '.npy', test_loader)

    def save_embedding(self, output, data_loader):
        all_embeddings = self.get_embedding(data_loader)
        np.save(output, all_embeddings)

    def get_embedding(self, data_loader):
        embedding_list = []
        self.model.eval()
        with torch.no_grad():
            for data in data_loader:
                building_feature, building_mask, xy, poi_feature, poi_mask, svi_emb, svi_mask = data
                embedding = self.forward(building_feature, building_mask, xy, poi_feature, poi_mask, svi_emb, svi_mask)
                embedding_list.append(embedding.detach().cpu().numpy())
        all_embeddings = np.concatenate(embedding_list, axis=0)
        self.model.train()
        return all_embeddings

    def get_embeddings(self):
        dataset = UnsupervisedPatternDataset(self.city_data)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False,
                                                  collate_fn=UnsupervisedPatternDataset.collate_fn)
        return self.get_embedding(test_loader)


class RegionTrainer(object):
    def __init__(self, city_data, pattern_model, pattern_optimizer, pattern_scheduler, region_model, region_optimizer,
                 region_scheduler, early_stopping=10, device=torch.device('cuda')):
        self.city_data = city_data

        self.patterns = self.city_data.patterns
        self.pattern_model = pattern_model
        self.pattern_optimizer = pattern_optimizer
        self.pattern_scheduler = pattern_scheduler

        self.regions = self.city_data.regions
        self.region_model = region_model
        self.region_optimizer = region_optimizer
        self.region_scheduler = region_scheduler

        self.early_stopping = early_stopping
        self.device = device

    def save_embedding_freeze(self, path, loader):
        self.pattern_model.eval()
        self.region_model.eval()
        embeddings = {}
        with torch.no_grad():
            for data in loader:
                for key, pattern in data:
                    tensor = torch.from_numpy(np.vstack(pattern)).to(self.device)
                    embeddings[key] = self.region_model.get_embedding(tensor).cpu().numpy()
        with open(path, 'wb') as f:
            pkl.dump(embeddings, f)

    def eval_per_epoch(self, loader, baseline_embeddings, raw_labels):
        # 保存所有状态
        rng_state = {
            'np': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'py_random': random.getstate()
        }
        self.region_model.eval()
        embeddings = {}
        with torch.no_grad():
            for data in loader:
                for key, pattern in data:
                    tensor = torch.from_numpy(np.vstack(pattern)).to(self.device)
                    embeddings[key] = self.region_model.get_embedding(tensor).cpu().numpy()
        # [average_l1, std_l1, average_kl_div, std_kl_div, average_cos, std_cos]
        result = land_use_inference(embeddings, baseline_embeddings, raw_labels, [0.6,0.2,0.2], 10)

        self.region_model.train()
        # 恢复 NumPy
        np.random.set_state(rng_state['np'])
        # 恢复 PyTorch
        torch.set_rng_state(rng_state['torch_cpu'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_state['torch_cuda'])
        # 恢复 Python random
        random.setstate(rng_state['py_random'])

        return result


    def train_region_triplet_freeze(self, epochs, embeddings, save_name, 
                                    adaptive=True, window_sizes=None, use_wandb=False,
                                    _lambda=100, first_epoch=20):
        if adaptive:
            criterion = adaptive_triplet_loss
        else:
            criterion = triplet_loss

        test_dataset = FreezePatternForwardDataset(embeddings, self.city_data)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False,
                                                  collate_fn=FreezePatternForwardDataset.collate_fn)
        save_path = 'embeddings/' + self.city_data.city + '/'

        print('Building pretraining dataset...')
        if window_sizes is None:
            train_datasets = [FreezePatternPretrainDataset(embeddings, self.city_data, window_size=3000)]
        else:
            train_datasets = [FreezePatternPretrainDataset(embeddings, self.city_data, window_size=window_size) for
                              window_size in window_sizes]
        train_loaders = [torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True,
                                                     collate_fn=FreezePatternPretrainDataset.collate_fn) for
                         train_dataset in train_datasets]

        baseline_path = 'baselines/{}_doc2vec_grid.pkl'.format('singapore')
        with open(baseline_path, 'rb') as f:
            baseline_embeddings = pkl.load(f)
        with open('data/processed/' + 'Singapore' + '/downstream_region.pkl', 'rb') as f:
            raw_labels = pkl.load(f)

        for epoch in range(1, epochs+1):
            train_losses = []
            self.region_model.train()
            train_loader = train_loaders[epoch % len(train_loaders)]
            with tqdm(train_loader, desc='Epoch {}'.format(epoch)) as tqdm_batch:
                for data in tqdm_batch:
                    self.region_optimizer.zero_grad()
                    regions = []
                    patterns = []
                    try:
                        # In rare cases, several huge regions are generated, which will cause OOM error
                        # Ignore these cases
                        # data: (anchor, positive, negative)
                        for pattern in data:
                            packed = np.vstack(pattern)
                            tensor = torch.from_numpy(packed).to(self.device)
                            patterns.append(tensor)
                            regions.append(self.region_model(tensor))
                        anchor = regions[0].mean(dim=0)
                        positive = regions[1].mean(dim=0)
                        negative = regions[2].mean(dim=0)
                        if adaptive:
                            loss = criterion(anchor, positive, negative, patterns[1], patterns[2], _lambda=_lambda)
                        else:
                            loss = criterion(anchor.unsqueeze(0), positive.unsqueeze(0), negative.unsqueeze(0))
                        loss.backward()
                    except RuntimeError as e:
                        print(e)
                        continue
                    clip_grad_norm_(self.region_model.parameters(), 1.0)
                    self.region_optimizer.step()
                    tqdm_batch.set_postfix(loss=loss.item())
                    train_losses.append(loss.item())
            if epoch > 0 and epoch % 10 == 0:
                self.save_embedding_freeze(save_path + save_name + str(epoch) + '.pkl', test_loader)
            print('Epoch {}, Tiplet Loss: {}'.format(epoch, np.mean(train_losses)))
            # # [average_l1, std_l1, average_kl_div, std_kl_div, average_cos, std_cos]
            eval_res = self.eval_per_epoch(test_loader, baseline_embeddings, raw_labels)
            if use_wandb:
                swanlab.log({
                    "Region-loss": np.mean(train_losses),
                    "l1": eval_res[0],
                    "kl_div": eval_res[2],
                    "cos": eval_res[4]
                }, step=epoch+first_epoch)
            self.region_scheduler.step()

