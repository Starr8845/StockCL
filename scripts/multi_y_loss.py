import torch as torch
import pandas as pd
import numpy as np

# 这里尝试做一个改进  在模型warm up一段时间后，仅拉近距离比较近的正样本
def get_positive_mask(gt, multi_y, device, threshold=0.2, use_multi_y=False):
    if use_multi_y:
        temp = torch.abs(torch.tensor(multi_y.reshape(256,1,11)-multi_y.reshape(1,256,11), device=device))
        weight = torch.tensor([0,0,0,0,0.125,0.25,0.5,1,0.5,0.25,0.125], device=device).reshape(1,1,11)
        sim = (temp*weight).sum(dim=2)
        mask_positives1 = (sim < 2)
        mask_positives2 = ((torch.abs(gt.sub(gt.T)) < 0.3))
        mask_positives = (mask_positives1 & mask_positives2).float().to(device)
    else:
        mask_positives = ((torch.abs(gt.sub(gt.T)) < threshold)).float().to(device)
    return mask_positives
        

def multi_y_contrast_loss(features, gt, tau, multi_y, use_multi_y=False, loss_weight="all1", 
                          multi_y_repres=None, threshold=0.2, last_epoch_threshold=None, 
                          sim=None, prior_mask = None, all_negative=False, cos=False, same_class=True):
    device = features.device
    batch_size = features.shape[0]
    gt = gt.contiguous().view(-1, 1)
    mask_positives = get_positive_mask(gt, multi_y, device, threshold, use_multi_y=use_multi_y)
    if prior_mask is not None:
        mask_positives = mask_positives * prior_mask
    if type(last_epoch_threshold)==float:
        mask_positives = torch.abs(gt.sub(gt.T)) < threshold
        mask_positives2 = torch.matmul(features, features.T).detach()>last_epoch_threshold
        mask_positives = (mask_positives & mask_positives2).float().to(device)
    if same_class:
        same_class = (torch.mm(gt, gt.T)>0).float().to(device)
        mask_positives = mask_positives * same_class # 需要把严格涨和跌的区分开
    if all_negative:
        mask_negatives = torch.ones_like(mask_positives, dtype=torch.float, device=device)
    else:
        mask_negatives = (torch.abs(gt.sub(gt.T)) > threshold+0.1).float().to(device)   # 这些样本作为negatives
    positive_num, negative_num = mask_positives.sum(), mask_negatives.sum()
    mask_neutral = mask_positives + mask_negatives
    if cos:
        anchor_dot_contrast = torch.div(pair_wise_cos(features,features),tau)
    else:
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), tau)
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()
    logits_mask = torch.scatter(
        torch.ones_like(mask_positives), 1,
        torch.arange(batch_size).view(-1, 1).to(device), 0) * mask_neutral
    mask_positives = mask_positives * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-20)
    
    if loss_weight=='all1':
        mean_log_prob_pos = (mask_positives * log_prob).sum(1) / (mask_positives.sum(1) + 1e-20)
    elif loss_weight=="param":
        sim = torch.exp(torch.mm(multi_y_repres, multi_y_repres.T))  # 参数化的weight
        mean_log_prob_pos = (mask_positives * log_prob* sim).sum(1) / ((mask_positives*sim).sum(1) + 1e-20)
    elif loss_weight=="rule_gaze":
        sim = -1*torch.log(torch.abs(gt.sub(gt.T))+1e-2)  # 根据相似度定义的权重        
        mean_log_prob_pos = (mask_positives * log_prob* sim).sum(1) / ((mask_positives*sim).sum(1) + 1e-20)
    elif loss_weight=="rule_proxy": # 使用高斯核来定义权重
        sim = torch.exp(-1*torch.square(gt.sub(gt.T))/2)  # 根据相似度定义的权重        
        mean_log_prob_pos = (mask_positives * log_prob* sim).sum(1) / ((mask_positives*sim).sum(1) + 1e-20)

    elif loss_weight=="x_sim":
        mean_log_prob_pos = (mask_positives * log_prob* sim).sum(1) / ((mask_positives*sim).sum(1) + 1e-20)
    
    loss = -1 * mean_log_prob_pos
    loss = loss.view(1, batch_size).mean()
    return loss, (positive_num, negative_num)


def multi_criterion_con_loss(features, gt, tau, loss_weight="all1"):
    con_loss = 0
    for threshold in [0.1,0.15,0.2,0.25,0.3]:
        con_loss += multi_y_contrast_loss(features, gt, tau, multi_y=None, use_multi_y=False, loss_weight=loss_weight, multi_y_repres=None, threshold=threshold)
    return con_loss/5


def multi_y_contrast_loss_new(feature, tau, multi_y, metric="IC"):
    # ff = feature @ feature.T / tau
    feature_norm = feature/feature.norm(dim=1)[:, None]
    ff = torch.mm(feature_norm, feature_norm.transpose(0,1)) / tau
    ff_max, _ = torch.max(ff, dim=1, keepdim=True)
    exp_ziT_zj_div_tau = torch.exp(ff - ff_max.detach())
    if metric=="IC":
        df = pd.DataFrame(multi_y.T)
        sim = torch.tensor(df.corr(method = 'pearson').values, device=feature.device) 
    elif metric=="weightedl1":
        temp = torch.abs(torch.tensor(multi_y.reshape(256,1,11)-multi_y.reshape(1,256,11), device=feature.device))
        # weight = torch.tensor([0,0,0,0.125,0.25,0.5,1,1,0.5,0.25,0.125],device=multi_y.device).reshape(1,1,11)
        weight = torch.tensor([0,0,0,0,0,0,0,1,0,0,0],device=feature.device).reshape(1,1,11)
        sim = (temp*weight).mean(dim=2)
        sim = torch.ones_like(sim, device=feature.device)-sim
    elif metric=="single_y":
        single_y = multi_y[:,-4].reshape(256,1)
        sim = torch.abs(torch.tensor(single_y-single_y.T, device=feature.device))
        sim = torch.ones_like(sim, device=feature.device)-sim
        sim = torch.clamp(sim, -1, 1)
    loss = cal_con_loss(sim, exp_ziT_zj_div_tau)
    return loss


def cal_con_loss(sim, exp_ziT_zj_div_tau):
    N = sim.shape[0]
    col_idxs = [[(i + j) % N for j in range(N)] for i in range(N)]

    sim_unsqueeze = sim.unsqueeze(2).expand(N, N, N)
    expand_sim = torch.stack([sim[:, col_idx] for col_idx in col_idxs], dim=2)
    expand_exp = torch.stack([exp_ziT_zj_div_tau[:, col_idx] for col_idx in col_idxs], dim=2)

    mask = (expand_sim < sim_unsqueeze) & ((sim_unsqueeze < 0) | (expand_sim > 0))
    sip_div_sij = expand_sim / sim_unsqueeze
    Sigma = torch.sum(((sim_unsqueeze > 0) * sip_div_sij + (sim_unsqueeze < 0) * (1 / sip_div_sij)) * expand_exp * mask, dim=2)
    numerator = exp_ziT_zj_div_tau * (sim > 0) + (sim < 0)
    L_zi_zj = -torch.abs(sim) * torch.log(numerator / (exp_ziT_zj_div_tau + Sigma) + 1e-20)
    L_zi_zj = L_zi_zj*((sim > 0) + (sim < 0)*0.5)
    return L_zi_zj.mean()


def cal_mask_pos(batch_y, sample_y):
    mask_positives1 = np.abs(batch_y.reshape(-1,1)-sample_y.reshape(1,-1)) < 0.2
    temp = batch_y.reshape(-1,1)/sample_y.reshape(1,-1) 
    mask_positives2 = (temp<1.67) & (temp>0.6)
    mask_positives2 = mask_positives2 | mask_positives1
    return mask_positives2


def sample_positive(batch_x, batch_y, x_train_values_sorted, y_train_values_sorted, batch_ranking, threshold=0.2, max_num=None):
    batch_size=batch_x.shape[0]
    total_num = x_train_values_sorted.shape[0]
    # mask_positives = np.abs(batch_y.reshape(-1,1)-batch_y.reshape(1,-1)) < threshold
    mask_positives = cal_mask_pos(batch_y, batch_y)
    exist_pos_num = mask_positives.sum(axis=1)
    if max_num is None:
        max_num = np.max(exist_pos_num)
    sample_low_index = np.where(-1*max_num+batch_ranking<0,0,-1*max_num+batch_ranking)
    sample_high_index = np.where(max_num+batch_ranking>=total_num,total_num-1,max_num+batch_ranking)
    final_y = np.copy(batch_y)
    sampled_index = np.array([],dtype=np.int32)
    for i in range(batch_size):
        if exist_pos_num[i]>=max_num:
            continue
        temp_random_index = np.random.randint(low=sample_low_index[i], high=sample_high_index[i],size=(max_num-exist_pos_num[i]))
        temp_random_index = np.setdiff1d(temp_random_index, sampled_index)# 不能有重复元素
        sampled_index = np.concatenate([sampled_index, temp_random_index])
        sampled_samples_y = y_train_values_sorted[temp_random_index]
        final_y = np.concatenate([sampled_samples_y, final_y])
        # mask_positives = np.abs(batch_y.reshape(-1,1)-final_y.reshape(1,-1)) < threshold
        mask_positives = cal_mask_pos(batch_y, final_y)
        exist_pos_num = mask_positives.sum(axis=1)
    
    final_x = np.concatenate([batch_x, x_train_values_sorted[sampled_index]])

    # mask_positives = np.abs(batch_y.reshape(-1,1)-final_y.reshape(1,-1)) < threshold
    # exist_pos_num = mask_positives.sum(axis=1)
    # print(np.sort(exist_pos_num))
    return final_x, final_y


def multi_y_contrast_loss_sampling(repres_batch, batch_y, repres_final, final_y, tau, loss_weight="all1", 
                          threshold=0.2,
                          sim=None, prior_mask = None):
    device = repres_batch.device
    batch_size = repres_batch.shape[0]
    batch_y = batch_y.contiguous().view(-1, 1)
    mask_positives = (torch.abs(batch_y.reshape(-1,1)-final_y.reshape(1,-1)) < threshold).float().to(device) 
    if prior_mask is not None:
        mask_positives = mask_positives * prior_mask
    
    same_class = (torch.mm(batch_y.reshape(-1,1),final_y.reshape(1,-1))>0).float().to(device)
    mask_positives = mask_positives * same_class # 需要把严格涨和跌的区分开
    mask_negatives = (torch.abs(batch_y.reshape(-1,1)-final_y.reshape(1,-1)) > threshold+0.1).float().to(device) 
    positive_num, negative_num = mask_positives.sum(), mask_negatives.sum()
    mask_neutral = mask_positives + mask_negatives
    anchor_dot_contrast = torch.div(torch.matmul(repres_batch, repres_final.T), tau)
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()
    logits_mask = torch.scatter(
        torch.ones_like(mask_positives), 1,
        torch.arange(batch_size).view(-1, 1).to(device), 0) * mask_neutral
    mask_positives = mask_positives * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-20)
    
    if loss_weight=='all1':
        mean_log_prob_pos = (mask_positives * log_prob).sum(1) / (mask_positives.sum(1) + 1e-20)
    elif loss_weight=="rule":
        sim = -1*torch.log(torch.abs(batch_y.reshape(-1,1)-final_y.reshape(1,-1))+1e-2)  # 根据相似度定义的权重        
        mean_log_prob_pos = (mask_positives * log_prob* sim).sum(1) / ((mask_positives*sim).sum(1) + 1e-20)
    elif loss_weight=="x_sim":
        mean_log_prob_pos = (mask_positives * log_prob* sim).sum(1) / ((mask_positives*sim).sum(1) + 1e-20)
    
    loss = -1 * mean_log_prob_pos
    loss = loss.view(1, batch_size).mean()
    return loss, (positive_num, negative_num)


def pair_wise_cos(a,b):
    a_norm = a/a.norm(dim=1)[:, None]
    b_norm = b/b.norm(dim=1)[:, None]
    res = torch.mm(a_norm, b_norm.transpose(0,1))
    return res
