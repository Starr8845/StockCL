# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from typing import Text, Union
import copy
from qlib.utils import get_or_create_path
from qlib.log import get_module_logger

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from qlib.workflow.record_temp import PortAnaRecord
from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from utils import string_format
import os
import pickle
from multi_y_loss import multi_y_contrast_loss, multi_y_contrast_loss_new, multi_criterion_con_loss, sample_positive, multi_y_contrast_loss_sampling
from base_models import MemoryModule, ALSTMModel, GRUModel, TCNModel, Transformer, HyperPredictor
from rankNcontrast import RnCLoss


class ALSTM(Model):
    """ALSTM Model

    Parameters
    ----------
    d_feat : int
        input dimension for each time step
    metric: str
        the evaluation metric used in early stop
    optimizer : str
        optimizer name
    GPU : int
        the GPU ID used for training
    """

    def __init__(
        self,
        d_feat=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        n_epochs=200,
        lr=0.001,
        metric="",
        batch_size=2000,
        early_stop=15,
        loss="mse",
        optimizer="adam",
        GPU=0,
        seed=None,
        use_y_model=False,
        use_sample_weighter=False,
        use_memory_net=False,
        base_model="ASLTM",
        data_region="cn",
        use_multi_predictor = False,
        prompt_num=256,
        threshold=0.2,
        **kwargs
    ):
        # Set logger.
        self.logger = get_module_logger("ALSTM")
        self.logger.info("ALSTM pytorch version...")

        # set hyper-parameters.
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed
        self.use_y_model = use_y_model
        self.use_sample_weighter = use_sample_weighter
        self.use_memory_net = use_memory_net
        self.data_region=data_region
        self.use_multi_predictor = use_multi_predictor
        self.threshold=threshold
        
        self.logger.info(
            "ALSTM parameters setting:"
            "\nd_feat : {}"
            "\nhidden_size : {}"
            "\nnum_layers : {}"
            "\ndropout : {}"
            "\nn_epochs : {}"
            "\nlr : {}"
            "\nmetric : {}"
            "\nbatch_size : {}"
            "\nearly_stop : {}"
            "\noptimizer : {}"
            "\nloss_type : {}"
            "\ndevice : {}"
            "\nuse_GPU : {}"
            "\nseed : {}"
            "\ny_model : {}"
            "\nsample_weighter : {}"
            "\nmemory_net : {}"
            .format(
                d_feat,
                hidden_size,
                num_layers,
                dropout,
                n_epochs,
                lr,
                metric,
                batch_size,
                early_stop,
                optimizer.lower(),
                loss,
                self.device,
                self.use_gpu,
                seed,
                use_y_model,
                use_sample_weighter,
                use_memory_net
            )
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        if base_model=="ALSTM":
            self.ALSTM_model = ALSTMModel(
                d_feat=self.d_feat,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
            ).to(self.device)
        elif base_model=="GRU":      
            self.ALSTM_model = GRUModel(
                d_feat=self.d_feat,
                hidden_size=self.hidden_size, 
                num_layers=self.num_layers, 
                dropout=self.dropout
            ).to(self.device)      
        elif base_model=="TCN":
            self.ALSTM_model = TCNModel(
                num_input=6, output_size=1, num_channels=[128]*5, kernel_size=3, dropout=0.5  # 这里单独设置dropout
            ).to(self.device)
        elif base_model=="Transformer":
            self.ALSTM_model = Transformer(
                d_feat=6, d_model=self.hidden_size, nhead=2, num_layers=2, dropout=0, device=self.device   # transformer 的dropout为0
            ).to(self.device)
        else:
            raise NotImplementedError("no {} base model!".format(base_model))
        self.logger.info("model:\n{:}".format(self.ALSTM_model))
        params = [{"params":self.ALSTM_model.parameters()}]
        
        if self.use_sample_weighter:
            self.sample_weighter = sampel_weighter(hidden_size=self.hidden_size*4).to(self.device)
            params.append({"params":self.sample_weighter.parameters()})
        if self.use_y_model:
            self.y_model = ALSTMModel(d_feat=1, hidden_size=16, num_layers=2, rnn_type="GRU", dropout=0.0).to(self.device)
            self.y_model.load_state_dict(torch.load("models/y_GRU.pt"))
            params.append({"params":self.y_model.parameters()})
        if self.use_memory_net:
            if base_model=="ALSTM":
                mem_hidden = self.hidden_size*2
            elif base_model=="GRU":      
                mem_hidden = self.hidden_size
            elif base_model=="TCN":
                mem_hidden = 128
            elif base_model=="Transformer":
                mem_hidden = self.hidden_size
            self.memory_net = MemoryModule(hidden_size=mem_hidden,prompt_num=prompt_num).to(self.device)  
            params.append({"params":self.memory_net.parameters()})
            if self.use_multi_predictor:
                self.pred_model = HyperPredictor(hidden_size=mem_hidden).to(self.device)
                params.append({"params":self.pred_model.parameters()})
        
        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(params, lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(params, lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        

    def load_model(self, path):
        self.ALSTM_model.load_state_dict(torch.load(path))
        self.fitted = True
        
    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        loss = (pred - label) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)

        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)

        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])

        raise ValueError("unknown metric `%s`" % self.metric)



    def train_epoch(self, epoch, x_train, y_train, y_con, y_con_config={"y_con_type":"multi_y_contrast_loss","use_multi_y":False, "loss_weight":"all1", "metric":"IC"}):
        print(y_con_config)
        # 这里加入对比学习损失  在每个batch内进行样本间对比
        x_train_values = x_train.values
        if self.data_region=="cn":
            y_train_values = (y_train.values[:,-4]).squeeze()
        else: # usa
            y_train_values = (y_train.values[:,-5]).squeeze()
        multi_y_values = y_train.values
        sort_index = np.argsort(y_train_values)
        x_train_values_sorted = x_train_values[sort_index]
        y_train_values_sorted = y_train_values[sort_index]
        ranking = np.argsort(sort_index) 
        

        self.ALSTM_model.train()

        indices = np.arange(len(x_train_values))
        np.random.shuffle(indices)
        additional_loss = 0
        tra_reg_loss=0
        tra_loss_constraint = 0
        positive_num, negative_num = 0, 0
        all_one_prototype_count=0
        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break
            feature = torch.from_numpy(x_train_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_train_values[indices[i : i + self.batch_size]]).float().to(self.device)
            batch_ranking = ranking[indices[i : i + self.batch_size]]
            repres = self.ALSTM_model.get_repre(feature)
            loss=0
            if self.use_memory_net:
                repres_memory, loss_constraint, same_proto_mask, attention = self.memory_net.memory_enhance(repres, get_attention=True)
                if (same_proto_mask.sum()/len(same_proto_mask)).item()==256:
                    all_one_prototype_count+=1
                loss+=loss_constraint
                tra_loss_constraint+=loss_constraint.item()
                if self.use_multi_predictor:
                    pred = self.pred_model(repres, repres_memory)
            if not self.use_multi_predictor:
                pred = self.ALSTM_model.predict(repres)
            loss += self.loss_fn(pred, label)
            tra_reg_loss+=loss.item()
            
            if y_con:
                multi_y = multi_y_values[indices[i : i + self.batch_size]]
                if y_con_config["y_con_type"]=="multi_y_contrast_loss":
                    if y_con_config['loss_weight']=="param":
                        multi_y_input = torch.from_numpy(multi_y[:,:-3]).float().to(self.device)
                        multi_y_repres = self.y_model.get_repre(multi_y_input)
                    else:
                        multi_y_repres=None
                    if y_con_config["loss_weight"]=="x_sim":
                        sim = self.sample_weighter(repres_memory if self.use_memory_net else repres)
                    else:
                        sim=None
                    y_con_loss, (positive_num_, negative_num_) = multi_y_contrast_loss(features=repres_memory if self.use_memory_net else repres, 
                                                gt=label, tau=0.1, 
                                                multi_y=multi_y, use_multi_y=y_con_config["use_multi_y"],  
                                                loss_weight=y_con_config["loss_weight"],
                                                all_negative= y_con_config["all_negative"],
                                                multi_y_repres=multi_y_repres, threshold=0.2, 
                                                sim=sim, prior_mask = same_proto_mask if self.use_memory_net else None)
                    positive_num+=positive_num_
                    negative_num+=negative_num_
                elif y_con_config["y_con_type"]=="multi_criterion_con_loss":
                    y_con_loss = multi_criterion_con_loss(features=repres, gt=label, tau=0.1, loss_weight=y_con_config["loss_weight"])
                elif y_con_config["y_con_type"]=="multi_y_contrast_loss_new":
                    y_con_loss = multi_y_contrast_loss_new(repres, tau=0.1, multi_y=multi_y, metric=y_con_config["metric"])
                elif y_con_config["y_con_type"]=="sampling":
                    final_x, final_y = sample_positive(batch_x=x_train_values[indices[i : i + self.batch_size]], 
                                                       batch_y=y_train_values[indices[i : i + self.batch_size]], 
                                                       x_train_values_sorted=x_train_values_sorted, 
                                                       y_train_values_sorted=y_train_values_sorted, 
                                                       batch_ranking=batch_ranking, 
                                                       threshold=self.threshold, 
                                                       max_num=None)
                    final_x, final_y = torch.from_numpy(final_x).float().to(self.device), torch.from_numpy(final_y).float().to(self.device)
                    repres_final = self.ALSTM_model.get_repre(final_x)
                    if self.use_memory_net:
                        repres_final_memory, loss_constraint, same_proto_mask, attention_final = self.memory_net.memory_enhance(repres_final, get_attention=True)
                        if (same_proto_mask.sum()/len(same_proto_mask)).item()==256:
                            all_one_prototype_count+=1
                        same_proto_mask = same_proto_mask[:len(feature), :]
                        attention_sim = torch.einsum("nd,Nd->nN", attention, attention_final)
                    
                    y_con_loss, (positive_num_, negative_num_) = multi_y_contrast_loss_sampling(repres_batch=repres_memory if self.use_memory_net else repres, 
                                                                batch_y=label, 
                                                                repres_final=repres_final_memory if self.use_memory_net else repres_final, 
                                                                final_y=final_y, 
                                                                tau=0.1, 
                                                                loss_weight="x_sim" if self.use_memory_net and epoch>5 else "all1",  # all1
                                                                threshold=0.2,
                                                                sim=attention_sim if self.use_memory_net and epoch>5 else None, 
                                                                prior_mask = same_proto_mask if self.use_memory_net and epoch>5 else None)
                    # y_con_loss, (positive_num_, negative_num_) = multi_y_contrast_loss_sampling(
                    #                                             repres_batch=repres, 
                    #                                             batch_y=label, 
                    #                                             repres_final=repres_final, 
                    #                                             final_y=final_y, 
                    #                                             tau=0.1, 
                    #                                             loss_weight="x_sim" if self.use_memory_net else "all1",  # all1
                    #                                             threshold=self.threshold,
                    #                                             sim=attention_sim if self.use_memory_net else None, 
                    #                                             prior_mask = same_proto_mask if self.use_memory_net else None)

                    positive_num+=positive_num_
                    negative_num+=negative_num_
                elif y_con_config["y_con_type"]=="rankN":
                    # 使用rankNcontrast方法
                    criterion = RnCLoss(temperature=2, label_diff='l1', feature_sim='l2')
                    temp_repres = repres.reshape(repres.shape[0],1,-1).expand(-1,2,-1)    # rankNcontrast要求做扰,这里设定扰动为identification function
                    label = label.reshape(-1,1)
                    y_con_loss = criterion(temp_repres, label)
                elif y_con_config['y_con_type']=="proxy":
                    # 即所有的样本对都当成正样本对，但是权重不一样
                    y_con_loss, (positive_num_, negative_num_) = multi_y_contrast_loss(
                                                features=repres, 
                                                gt=label, tau=2, 
                                                multi_y=multi_y, use_multi_y=False,  
                                                loss_weight="rule_proxy",
                                                threshold=100, 
                                                all_negative=True, cos=True, same_class=False)
                    positive_num+=positive_num_
                    negative_num+=negative_num_
                elif y_con_config['y_con_type']=="gaze":
                    # 即所有的样本对都当成正样本对，但是权重不一样
                    y_con_loss, (positive_num_, negative_num_) = multi_y_contrast_loss(
                                                features=repres, 
                                                gt=label, tau=2, 
                                                multi_y=multi_y, use_multi_y=False,  
                                                loss_weight="rule_gaze",
                                                threshold=100, 
                                                all_negative=True, cos=True, same_class=False)
                    positive_num+=positive_num_
                    negative_num+=negative_num_
                elif y_con_config["y_con_type"]=="feature_sim_con":
                    feature_sim = torch.matmul(repres, repres.T).detach()
                    _, rank_indices = torch.sort(feature_sim,dim=1)
                    _, rank_indices = torch.sort(rank_indices,dim=1)
                    n = repres.shape[0]
                    ratio=y_con_config["ratio"]  # 只取前10%最像的
                    mask = rank_indices> n*(1-ratio)
                    y_con_loss, (positive_num_, negative_num_) = multi_y_contrast_loss(repres, 
                                                gt=label, tau=0.1, 
                                                multi_y=multi_y, use_multi_y=False,  
                                                loss_weight="all1",
                                                threshold=0.2, 
                                                prior_mask = mask)
                    positive_num+=positive_num_
                    negative_num+=negative_num_

                if "lam" in y_con_config:
                    loss += y_con_loss*y_con_config["lam"]     
                    additional_loss+=y_con_loss.item()*y_con_config["lam"]
                else:
                    loss += y_con_loss/5     
                    additional_loss+=y_con_loss.item()/5
            
            
            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.ALSTM_model.parameters(), 3.0)
            self.train_optimizer.step()

        positive_num/=len(x_train_values)
        negative_num/=len(x_train_values)
        tra_reg_loss, additional_loss, tra_loss_constraint = tra_reg_loss/(len(indices)//self.batch_size+1), additional_loss/(len(indices)//self.batch_size+1),tra_loss_constraint/(len(indices)//self.batch_size+1)
        addtional_train_info={"all same batch num":all_one_prototype_count,"positive_num":positive_num, "negative_num":negative_num}
        return tra_reg_loss, additional_loss, tra_loss_constraint, addtional_train_info

    def cal_my_IC(self, prediction, label):
        pd_label = pd.Series(label.squeeze())
        pd_prediction = pd.Series(prediction.squeeze())
        return pd_prediction.corr(pd_label), pd_prediction.corr(pd_label, method="spearman")

    def test_epoch(self, data_x, data_y, data_split):  
        x_values = data_x.values
        if self.data_region=="cn":
            y_values = (data_y.values[:,-4]).squeeze()
        else: # usa
            y_values = (data_y.values[:,-5]).squeeze()

        self.ALSTM_model.eval()
        if self.use_memory_net:
            self.memory_net.eval()
        if self.use_multi_predictor:
            self.pred_model.eval()
        preds = []
        scores = []
        losses = []
        my_ICs = []
        my_RICs = []

        indices = np.arange(len(x_values))

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_values[indices[i : i + self.batch_size]]).float().to(self.device)

            with torch.no_grad():
                repres = self.ALSTM_model.get_repre(feature)
                if self.use_memory_net:
                    repres_memory, loss_constraint, same_proto_mask, attention = self.memory_net.memory_enhance(repres, get_attention=True)
                    if self.use_multi_predictor:
                        pred = self.pred_model(repres, repres_memory)

                if not self.use_multi_predictor:
                    pred = self.ALSTM_model.predict(repres)
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())
                preds.append(pred.detach().cpu().numpy().squeeze())
                score = self.metric_fn(pred, label)
                scores.append(score.item())
            
        preds = np.concatenate(preds)
        daily_index, daily_count = data_split
        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            pred = preds[batch]
            label = y_values[batch]
            IC, RIC = self.cal_my_IC(pred, label)
            my_ICs.append(IC)
            my_RICs.append(RIC)
        ICs = {"IC":np.nanmean(my_ICs),"ICIR":np.nanmean(my_ICs)/np.nanstd(my_ICs),"rankIC":np.nanmean(my_RICs),"rankICIR":np.nanmean(my_RICs)/np.nanstd(my_RICs)}
        return np.mean(losses), np.mean(scores), ICs
    
    def get_daily_inter(self, df, shuffle=False):
        # organize the train data into daily batches
        daily_count = df.groupby(level=0).size().values
        daily_index = np.roll(np.cumsum(daily_count), 1)
        daily_index[0] = 0
        if shuffle:
            # shuffle data
            daily_shuffle = list(zip(daily_index, daily_count))
            np.random.shuffle(daily_shuffle)
            daily_index, daily_count = zip(*daily_shuffle)
        return daily_index, daily_count

    def ana_eval(self, sr, recorder, x_test, benchmark):
        index = x_test.index
        self.ALSTM_model.eval()
        x_values = x_test.values
        sample_num = x_values.shape[0]
        preds = []

        for begin in range(sample_num)[:: self.batch_size]:
            if sample_num - begin < self.batch_size:
                end = sample_num
            else:
                end = begin + self.batch_size

            x_batch = torch.from_numpy(x_values[begin:end]).float().to(self.device)

            repres = self.ALSTM_model.get_repre(x_batch)
            with torch.no_grad():
                if self.use_memory_net:
                    repres_memory, loss_constraint, same_proto_mask, attention = self.memory_net.memory_enhance(repres, get_attention=True)
                    if self.use_multi_predictor:
                        pred = self.pred_model(repres, repres_memory)
                if not self.use_multi_predictor:
                    pred = self.ALSTM_model.predict(repres)
                pred = pred.detach().cpu().numpy()

            preds.append(pred.reshape(-1))

        sr.save_pred(pd.Series(np.concatenate(preds), index=index))
        backtest_config = {
                "strategy": {
                    "class": "TopkDropoutStrategy",
                    "module_path": "qlib.contrib.strategy",
                    "kwargs": {"signal": "<PRED>", "topk": 50, "n_drop": 5},
                },
                "backtest": {
                    "start_time": None,
                    "end_time": None,
                    "account": 100000000,
                    "benchmark": benchmark,
                    "exchange_kwargs": {
                        "limit_threshold": 0.095 if self.data_region=="cn" else None,
                        "deal_price": "close",
                        "open_cost": 0.0005,
                        "close_cost": 0.0015,
                        "min_cost": 5,
                    },
                },
            }
        par = PortAnaRecord(recorder, backtest_config, "day")
        artifact_objects = par.generate()
        return_withcost = artifact_objects['port_analysis_1day.pkl'].loc[("excess_return_with_cost","annualized_return"),"risk"]
        information_ratio_withcost = artifact_objects['port_analysis_1day.pkl'].loc[("excess_return_with_cost","information_ratio"),"risk"]
        max_drawdown = artifact_objects['port_analysis_1day.pkl'].loc[("excess_return_with_cost","max_drawdown"),"risk"]
        return {"return_withcost":return_withcost, "information_ratio_withcost":information_ratio_withcost, "max_drawdown":max_drawdown}

    def fit(
        self,
        dataset: DatasetH,
        recorder,
        evals_result=dict(),
        save_path=None,
        y_con=False,
        y_con_config={},
        sr=None,
        benchmark="all"
    ):
        df_train, df_valid = dataset.prepare(
            ["train", "valid"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        df_test, = dataset.prepare(
            ["test"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_I,
        )
        # df_test = df_test.dropna(axis=0, how='any')
        df_train = df_train.dropna(axis=0, how='any',subset=[('label','LABEL-2')])
        df_valid = df_valid.dropna(axis=0, how='any',subset=[('label','LABEL-2')])
        df_test = df_test.dropna(axis=0, how='any',subset=[('label','LABEL-2')])
        df_train = df_train.fillna(0)
        df_valid = df_valid.fillna(0)
        df_test = df_test.fillna(0)
        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]
        x_test, y_test = df_test["feature"], df_test["label"]
        train_split = self.get_daily_inter(df_train, shuffle=False)
        valid_split = self.get_daily_inter(df_valid, shuffle=False)
        test_split = self.get_daily_inter(df_test, shuffle=False)
        # save_path = get_or_create_path(save_path)
        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self.fitted = True

        # temp
        train_loss, train_score, train_ICs = self.test_epoch(x_train, y_train, train_split)
        val_loss, val_score, valid_ICs = self.test_epoch(x_valid, y_valid, valid_split)
        test_loss, test_score, test_ICs = self.test_epoch(x_test, y_test, test_split)
        self.logger.info("train %.5f, valid %.6f, test %.5f" % (train_score, val_score, test_score))
        self.logger.info(f"train IC:{string_format(train_ICs)}")
        self.logger.info(f"valid IC:{string_format(valid_ICs)}")
        self.logger.info(f"test IC:{string_format(test_ICs)}")
        # if sr is not None:
        #     ana_eval_result = self.ana_eval(sr, recorder, x_test, benchmark)
        #     self.logger.info(f"ana_eval_result:{string_format(ana_eval_result)}")
        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            tra_loss, additional_loss, prototype_constraint_loss, addtional_train_info = self.train_epoch(step, x_train, y_train, y_con=y_con, y_con_config=y_con_config)
            train_loss, train_score, train_ICs = self.test_epoch(x_train, y_train, train_split)
            val_loss, val_score, valid_ICs = self.test_epoch(x_valid, y_valid, valid_split)
            test_loss, test_score, test_ICs = self.test_epoch(x_test, y_test, test_split)
            self.logger.info("train_reg %.5f, additional_loss %.5f, prototype_constraint_loss %.3f" % (tra_loss, additional_loss, prototype_constraint_loss))
            self.logger.info(f"additional train info: {string_format(addtional_train_info)}")
            self.logger.info("train %.5f, valid %.6f, test %.5f" % (train_score, val_score, test_score))
            self.logger.info(f"train IC:{string_format(train_ICs)}")
            self.logger.info(f"valid IC:{string_format(valid_ICs)}")
            self.logger.info(f"test IC:{string_format(test_ICs)}")
            if sr is not None:
                ana_eval_result = self.ana_eval(sr, recorder, x_test, benchmark)
                self.logger.info(f"ana_eval_result:{string_format(ana_eval_result)}")
            recorder.log_metrics(step=step,train_score=train_score, val_score=val_score, test_score=test_score, 
                                 **{'train '+key:train_ICs[key] for key in train_ICs}, 
                                 **{'valid '+key:valid_ICs[key] for key in valid_ICs},
                                 **{'test '+key:test_ICs[key] for key in test_ICs}, 
                                 tra_loss=tra_loss, additional_loss=additional_loss,
                                 prototype_constraint_loss=prototype_constraint_loss,
                                 **addtional_train_info,
                                 **(ana_eval_result if sr is not None else {})
                                 )
            
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if valid_ICs['IC'] > best_score:
                best_score = valid_ICs['IC']
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.ALSTM_model.state_dict())
                if self.use_memory_net:
                    best_param_memory = copy.deepcopy(self.memory_net.state_dict())
                    best_param_predictor = copy.deepcopy(self.memory_net.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break
            
            # torch.save(self.ALSTM_model.state_dict(), f"{save_path}/ALSTM_{step}.pt")
            # if self.use_memory_net:
            #     torch.save(self.memory_net.state_dict(), f"{save_path}/memory_{step}.pt")

        self.logger.info("best valid IC: %.6f @ %d" % (best_score, best_epoch))
        self.ALSTM_model.load_state_dict(best_param)
        torch.save(best_param, f"{save_path}/ALSTM.pt")
        if self.use_memory_net:
            torch.save(best_param_memory, f"{save_path}/memory.pt")
        if self.use_multi_predictor:
            torch.save(best_param_predictor, f"{save_path}/best_param_predictor.pt")
        self.logger.info(f"model saved: {save_path}")
        if self.use_gpu:
            torch.cuda.empty_cache()
    
    def sample_inference(self, data_x, data_y):  
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)
        self.ALSTM_model.eval()
        indices = np.arange(len(x_values))
        for i in range(len(indices))[:: self.batch_size]:
            feature = torch.from_numpy(x_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = y_values[indices[i : i + self.batch_size]]
            with torch.no_grad():
                repres = self.ALSTM_model.get_repre(feature)
                pred = self.ALSTM_model.predict(repres)
                return feature.detach().cpu().numpy(), repres.detach().cpu().numpy(), label, repres.cpu().numpy(), pred.detach().cpu().numpy()

    def get_sampled_repre(self, dataset, name):
        path = f"./repres/{name}"
        if not os.path.exists(path):
            os.mkdir(path)

        df_train, df_valid, df_test = dataset.prepare(
            ["train", "valid", "test"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        df_test, = dataset.prepare(
            ["test"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_I,
        )
        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]
        x_test, y_test = df_test["feature"], df_test["label"]
        train_repre, valid_repre, test_repre = self.sample_inference(x_train, y_train), self.sample_inference(x_valid, y_valid), self.sample_inference(x_test, y_test)
        with open(f"{path}/train_repre.pkl",'wb') as f:
            pickle.dump(train_repre, f)
        with open(f"{path}/valid_repre.pkl",'wb') as f:
            pickle.dump(valid_repre, f)
        with open(f"{path}/test_repre.pkl",'wb') as f:
            pickle.dump(test_repre, f)

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        x_test = x_test.fillna(0)
        index = x_test.index
        self.ALSTM_model.eval()
        x_values = x_test.values
        sample_num = x_values.shape[0]
        preds = []

        for begin in range(sample_num)[:: self.batch_size]:
            if sample_num - begin < self.batch_size:
                end = sample_num
            else:
                end = begin + self.batch_size

            x_batch = torch.from_numpy(x_values[begin:end]).float().to(self.device)

            with torch.no_grad():
                repres = self.ALSTM_model.get_repre(x_batch)
                if self.use_memory_net:
                    repres_memory, loss_constraint, same_proto_mask, attention = self.memory_net.memory_enhance(repres, get_attention=True)
                    if self.use_multi_predictor:
                        pred = self.pred_model(repres, repres_memory)
                if not self.use_multi_predictor:
                    pred = self.ALSTM_model.predict(repres)
                pred = pred.detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)




class sampel_weighter(nn.Module):  
    def __init__(self, hidden_size=128):
        super().__init__()
        self.weight_left = nn.Parameter(torch.FloatTensor(size=(1, hidden_size)))
        self.weight_right = nn.Parameter(torch.FloatTensor(size=(1, hidden_size)))
        nn.init.xavier_normal_(self.weight_left, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_normal_(self.weight_right, gain=nn.init.calculate_gain("relu"))
    
    def forward(self, repres):
        # repres: [N,d]
        left = torch.mm(self.weight_left, repres.T) # 1*N
        right = torch.mm(self.weight_right, repres.T) # 1*N
        sim = left + right.T # N*N  # 可以使用2*sigmoid-1映射到-1到1之间
        return sim
    




