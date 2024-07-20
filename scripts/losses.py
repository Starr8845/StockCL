import torch as torch
import torch.nn as nn
import numpy as np



def y_contrast_loss(features, gt, temp, threshold1, threshold2, threshold_cluster=0, use_loss_weight=False):
    device = features.device
    batch_size = features.shape[0]
    gt = gt.contiguous().view(-1, 1)
    mask_positives = ((torch.abs(gt.sub(gt.T)) < threshold1)).float().to(device)    # 这里考虑再乘一下 hard samples的mask   # 要拉近的是hard samples的距离
    same_class = (torch.mm(gt, gt.T)>threshold_cluster).float().to(device)
    mask_positives = mask_positives * same_class # 需要把严格涨和跌的区分开
    mask_negatives = (torch.abs(gt.sub(gt.T)) > threshold2).float().to(device)   # 这些样本作为negatives?
    # print(f"mask_positives:{mask_positives.sum()},mask_negatives:{mask_negatives.sum()}")
    mask_neutral = mask_positives + mask_negatives
    anchor_dot_contrast = torch.div(torch.matmul(features, features.T), temp)
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()
    logits_mask = torch.scatter(
        torch.ones_like(mask_positives), 1,
        torch.arange(batch_size).view(-1, 1).to(device), 0) * mask_neutral
    mask_positives = mask_positives * logits_mask
    exp_logits = torch.exp(logits) * logits_mask

    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-20)
    
    if use_loss_weight:
        sim = -1*torch.log(torch.abs(gt.sub(gt.T))+1e-2) # 根据相似度定义的权重
        mean_log_prob_pos = (mask_positives * log_prob* sim).sum(1) / ((mask_positives*sim).sum(1) + 1e-20)
    else:
        mean_log_prob_pos = (mask_positives * log_prob).sum(1) / (mask_positives.sum(1) + 1e-20)
    
    loss = -1 * mean_log_prob_pos
    loss = loss.view(1, batch_size).mean()
    return loss


def self_contrastive_loss(features, features_augmented, gt, temp):
    anchor_dot_contrast = torch.div(torch.matmul(features, features_augmented.T), temp)
    # anchor_dot_contrast = pair_wise_cos(features, features_augmented) / temp
    # 这里需要做normlization
    # print(anchor_dot_contrast.shape)
    # print(anchor_dot_contrast)
    batch_size = features.shape[0]
    gt = gt.contiguous().view(-1, 1)
    mask_negatives = ((torch.abs(gt.sub(gt.T)) > 0.3)).to(features.device) # mask_negatives这些样本 才应该作为负样本存在
    diagnal = torch.scatter(
        torch.zeros_like(mask_negatives, device=features.device), 1, 
        torch.arange(batch_size, device=features.device).view(-1, 1), 1).type(torch.BoolTensor).to(features.device)
    mask_negatives = mask_negatives | diagnal
    anchor_dot_contrast[~mask_negatives] = -100
    # anchor_dot_contrast[]
    criterion = nn.CrossEntropyLoss()
    targets = torch.arange(features.size(0)).to(features.device)
    loss = criterion(anchor_dot_contrast, targets)
    return loss


def pair_wise_cos(a,b):
    a_norm = a/a.norm(dim=1)[:, None]
    b_norm = b/b.norm(dim=1)[:, None]
    res = torch.mm(a_norm, b_norm.transpose(0,1))
    return res


def get_imbalance_spilit(min_num, max_num, split):
    delta = (max_num - min_num)/((1+split/2)*(split/2))
    boundary = np.zeros(split+1)
    boundary[int(split/2)] = (min_num+max_num)/2
    for i in range(1,int(split/2)+1):
        # boundary[int(split/2)+i] = boundary[int(split/2)+i-1] + i*delta
        # boundary[int(split/2)-i] = boundary[int(split/2)-i+1] - i*delta
        boundary[int(split/2)+i] = boundary[int(split/2)+i-1] + (int(split/2)-i+1)*delta
        boundary[int(split/2)-i] = boundary[int(split/2)-i+1] - (int(split/2)-i+1)*delta
    return boundary


def PCLloss(features, gt, temp, split_num, uniform_split=True):
    batch_size = features.shape[0]
    device = features.device
    gt_max = torch.max(gt)
    gt_min = torch.min(gt)
    if uniform_split:
        boundary = torch.linspace(gt_min, gt_max, split_num+1)
    else:
        boundary = get_imbalance_spilit(gt_min, gt_max, split_num)  # 两侧gap大，中间gap小
    mask = torch.zeros((split_num, batch_size), dtype=torch.bool).to(device)
    for i in range(split_num):
        mask[i,:]= (gt>=boundary[i]) & (gt<=boundary[i+1])
    mask = mask.float()
    protypes = torch.div(mask @ features, temp)
    protypes = protypes/(mask.sum(1).reshape(-1,1))
    target = (mask.argmax(0)).to(device)
    logits = features @ protypes.T
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, target)
    return loss

