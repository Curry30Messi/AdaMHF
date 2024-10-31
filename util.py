import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler

def collate_mil_survival(batch):
    images = torch.cat([item[0] for item in batch], dim=0)
    omics = torch.cat([item[1] for item in batch], dim=0).type(torch.FloatTensor)
    labels = torch.LongTensor([int(item[2]) for item in batch])
    event_times = np.array([item[3] for item in batch])
    censored = torch.FloatTensor([int(item[4]) for item in batch])
    return [images, omics, labels, event_times, censored]

def collate_mil_survival_cluster(batch):
    images = torch.cat([item[0] for item in batch], dim=0)
    cluster_ids = torch.cat([item[1] for item in batch], dim=0).type(torch.LongTensor)
    omics = torch.cat([item[2] for item in batch], dim=0).type(torch.FloatTensor)
    labels = torch.LongTensor([int(item[3]) for item in batch])
    event_times = np.array([item[4] for item in batch])
    censored = torch.FloatTensor([int(item[5]) for item in batch])
    return [images, cluster_ids, omics, labels, event_times, censored]

def collate_mil_survival_sig(batch):
    images = torch.cat([item[0] for item in batch], dim=0)
    omics_list = [torch.cat([item[i] for item in batch], dim=0).type(torch.FloatTensor) for i in range(1, 7)]
    labels = torch.LongTensor([int(item[7]) for item in batch])
    event_times = np.array([item[8] for item in batch])
    censored = torch.FloatTensor([int(item[9]) for item in batch])
    return [images] + omics_list + [labels, event_times, censored]

def make_balanced_weights(dataset):
    total_samples = float(len(dataset))
    class_weights = [total_samples / len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]
    sample_weights = [0] * int(total_samples)
    for index in range(len(dataset)):
        label = dataset.getlabel(index)
        sample_weights[index] = class_weights[label]
    return torch.DoubleTensor(sample_weights)

class SubsetSequentialSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def get_data_loader(split_dataset, is_training=False, is_testing=False, use_weighted_sampling=False, modality='coattn', batch_size=1):
    if modality == 'coattn':
        collate_fn = collate_mil_survival_sig
    elif modality == 'cluster':
        collate_fn = collate_mil_survival_cluster
    else:
        collate_fn = collate_mil_survival

    worker_args = {'num_workers': 0} if torch.cuda.is_available() else {}
    if not is_testing:
        if is_training:
            if use_weighted_sampling:
                weights = make_balanced_weights(split_dataset)
                loader = DataLoader(split_dataset, batch_size=batch_size, sampler=WeightedRandomSampler(weights, len(weights)), collate_fn=collate_fn, **worker_args)
            else:
                loader = DataLoader(split_dataset, batch_size=batch_size, sampler=RandomSampler(split_dataset), collate_fn=collate_fn, **worker_args)
        else:
            loader = DataLoader(split_dataset, batch_size=batch_size, sampler=SequentialSampler(split_dataset), collate_fn=collate_fn, **worker_args)
    else:
        sample_ids = np.random.choice(np.arange(len(split_dataset)), int(len(split_dataset) * 0.1), replace=False)
        loader = DataLoader(split_dataset, batch_size=1, sampler=SubsetSequentialSampler(sample_ids), collate_fn=collate_fn, **worker_args)

    return loader

def set_random_seed(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


import math
from einops import rearrange, reduce
from typing import Optional
import warnings

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from torch.nn.parameter import Parameter
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear as _LinearWithBias
from torch import Tensor
from torch.overrides import has_torch_function, handle_torch_function


def init_weights(module):
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                layer.bias.data.zero_()
        if isinstance(layer, nn.LayerNorm):
            nn.init.constant_(layer.bias, 0)
            nn.init.constant_(layer.weight, 1.0)


class BilinearFusion(nn.Module):
    def __init__(
            self,
            skip=0,
            use_bilinear=0,
            gate1=1,
            gate2=1,
            dim1=128,
            dim2=128,
            scale_dim1=1,
            scale_dim2=1,
            mmhid=256,
            dropout_rate=0.25,
    ):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1 // scale_dim1, dim2 // scale_dim2
        skip_dim = dim1_og + dim2_og if skip else 0

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = (
            nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og + dim2_og, dim1))
        )
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = (
            nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og + dim2_og, dim2))
        )
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(nn.Linear((dim1 + 1) * (dim2 + 1), 256), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(256 + skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))

    def forward(self, vec1, vec2):
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1) * h1)
        else:
            h1 = self.linear_h1(vec1)
            o1 = self.linear_o1(h1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2) if self.use_bilinear else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2) * h2)
        else:
            h2 = self.linear_h2(vec2)
            o2 = self.linear_o2(h2)

        o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
        o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        if self.skip:
            out = torch.cat((out, vec1, vec2), 1)
        out = self.encoder2(out)
        return out


def SNN_Block(dim1, dim2, dropout=0.25):
    return nn.Sequential(nn.Linear(dim1, dim2), nn.ELU(), nn.AlphaDropout(p=dropout))


def initialize_max_weights(module):
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            stdv = 1.0 / math.sqrt(layer.weight.size(1))
            layer.weight.data.normal_(0, stdv)
            layer.bias.data.zero_()


def exists(val):
    return val is not None


def moore_penrose_pinv(x, iters=6):
    device = x.device
    abs_x = torch.abs(x)
    col_sum = abs_x.sum(dim=-1)
    row_sum = abs_x.sum(dim=-2)
    z = rearrange(x, "... i j -> ... j i") / (torch.max(col_sum) * torch.max(row_sum))

    identity = torch.eye(x.shape[-1], device=device)
    identity = rearrange(identity, "i j -> () i j")

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * identity - (xz @ (15 * identity - (xz @ (7 * identity - xz)))))

    return z


class NystromAttention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_landmarks=256,
            pinv_iterations=6,
            residual=True,
            residual_conv_kernel=33,
            eps=1e-8,
            dropout=0.0,
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head
        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.residual = residual
        if residual:
            self.res_conv = nn.Conv2d(heads, heads, (residual_conv_kernel, 1), padding=(residual_conv_kernel // 2, 0), groups=heads, bias=False)

    def forward(self, x, mask=None, return_attn=False):
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps
        remainder = n % m
        if remainder > 0:
            padding = m - remainder
            x = F.pad(x, (0, 0, padding, 0), value=0)
            if exists(mask):
                mask = F.pad(mask, (padding, 0), value=False)

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        if exists(mask):
            mask = rearrange(mask, "b n -> b () n")
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))
        q = q * self.scale
        l = ceil(n / m)
        q_landmarks = reduce(q, "... (n l) d -> ... n d", "sum", l=l)
        k_landmarks = reduce(k, "... (n l) d -> ... n d", "sum", l=l)
        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, "... (n l) -> ... n", "sum", l=l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0
        q_landmarks /= divisor
        k_landmarks /= divisor
        sim1 = einsum("... i d, ... j d -> ... i j", q, k_landmarks)
        sim2 = einsum("... i d, ... j d -> ... i j", q_landmarks, k_landmarks)
        sim3 = einsum("... i d, ... j d -> ... i j", q_landmarks, k)
        if exists(mask):
            mask_value = -torch.finfo(q.dtype).max
            sim1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)
        attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_pinv(attn2, iters)
        out = (attn1 @ attn2_inv) @ (attn3 @ v)
        if self.residual:
            out += self.res_conv(v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        out = self.to_out(out)
        out = out[:, -n:]
        if return_attn:
            attn = attn1 @ attn2_inv @ attn3
            return out, attn
        return out
