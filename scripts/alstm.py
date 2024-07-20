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

from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from utils import string_format
from losses import y_contrast_loss, self_contrastive_loss, PCLloss
import os
import pickle


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
            "\nseed : {}".format(
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
            )
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.ALSTM_model = ALSTMModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        self.logger.info("model:\n{:}".format(self.ALSTM_model))
        # self.logger.info("model size: {:.4f} MB".format(count_parameters(self.ALSTM_model)))

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.ALSTM_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.ALSTM_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.ALSTM_model.to(self.device)
    
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

    def train_epoch(self, x_train, y_train, y_con, y_PCL, y_con_config={}, augmentation_dropout=False):
        # 这里加入对比学习损失  在每个batch内进行样本间对比
        x_train_values = x_train.values
        y_train_values = np.squeeze(y_train.values)

        self.ALSTM_model.train()

        indices = np.arange(len(x_train_values))
        np.random.shuffle(indices)
        additional_loss = 0
        tra_reg_loss=0
        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_train_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_train_values[indices[i : i + self.batch_size]]).float().to(self.device)
            
            # pred = self.ALSTM_model(feature)
            repres = self.ALSTM_model.get_repre(feature)
            pred = self.ALSTM_model.predict(repres)
            loss = self.loss_fn(pred, label)
            tra_reg_loss+=loss.item()
            if y_con:
                y_con_loss = y_contrast_loss(repres, gt=label, **y_con_config)
                loss += y_con_loss/50
                additional_loss+=y_con_loss.item()/50
            if augmentation_dropout:
                repres2 = self.ALSTM_model.get_repre(feature)
                self_con_loss = self_contrastive_loss(repres, repres2, gt=label, temp=0.1)
                loss += self_con_loss/50
                additional_loss += self_con_loss.item()/50
            if y_PCL:
                y_PCL_loss = PCLloss(features=repres, gt=label, temp=0.01, split_num=30, uniform_split=True)
                loss += y_PCL_loss/50
                additional_loss += y_PCL_loss.item()/50
            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.ALSTM_model.parameters(), 3.0)
            self.train_optimizer.step()
        return tra_reg_loss/(len(indices)//self.batch_size+1), additional_loss/(len(indices)//self.batch_size+1)

    def cal_my_IC(self, prediction, label):
        pd_label = pd.Series(label.squeeze())
        pd_prediction = pd.Series(prediction.squeeze())
        return pd_prediction.corr(pd_label), pd_prediction.corr(pd_label, method="spearman")

    def test_epoch(self, data_x, data_y, data_split):  
        # 增加一下IC的计算
        # prepare training data
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)

        self.ALSTM_model.eval()
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
                pred = self.ALSTM_model(feature)
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

    def fit(
        self,
        dataset: DatasetH,
        recorder,
        evals_result=dict(),
        save_path=None,
        y_con=False,
        y_con_config={}
    ):
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
        train_split = self.get_daily_inter(df_train, shuffle=False)
        valid_split = self.get_daily_inter(df_valid, shuffle=False)
        test_split = self.get_daily_inter(df_test, shuffle=False)
        save_path = get_or_create_path(save_path)
        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            tra_loss, additional_loss = self.train_epoch(x_train, y_train, y_con=y_con, y_con_config=y_con_config, augmentation_dropout=False, y_PCL=False)
            train_loss, train_score, train_ICs = self.test_epoch(x_train, y_train, train_split)
            val_loss, val_score, valid_ICs = self.test_epoch(x_valid, y_valid, valid_split)
            test_loss, test_score, test_ICs = self.test_epoch(x_test, y_test, test_split)
            self.logger.info("train_reg %.5f, additional_loss %.5f" % (tra_loss, additional_loss))
            self.logger.info("train %.5f, valid %.6f, test %.5f" % (train_score, val_score, test_score))
            self.logger.info(f"train IC:{string_format(train_ICs)}")
            self.logger.info(f"valid IC:{string_format(valid_ICs)}")
            self.logger.info(f"test IC:{string_format(test_ICs)}")
            recorder.log_metrics(step=step,train_score=train_score, val_score=val_score, test_score=test_score, 
                                 **{'train '+key:train_ICs[key] for key in train_ICs}, 
                                 **{'valid '+key:valid_ICs[key] for key in valid_ICs},
                                 **{'test '+key:test_ICs[key] for key in test_ICs}, 
                                 tra_loss=tra_loss, additional_loss=additional_loss)
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if valid_ICs['IC'] > best_score:
                best_score = valid_ICs['IC']
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.ALSTM_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best valid IC: %.6f @ %d" % (best_score, best_epoch))
        self.ALSTM_model.load_state_dict(best_param)
        torch.save(best_param, save_path)
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

    def get_sampled_repre(self, dataset, path):

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
                pred = self.ALSTM_model(x_batch).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)


class ALSTMModel(nn.Module):  # 实现的是attention LSTM 不是self attention
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, rnn_type="GRU"):
        super().__init__()
        self.hid_size = hidden_size
        self.input_size = d_feat
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.rnn_layer = num_layers
        self._build_model()

    def _build_model(self):
        try:
            klass = getattr(nn, self.rnn_type.upper())
        except Exception as e:
            raise ValueError("unknown rnn_type `%s`" % self.rnn_type) from e
        self.net = nn.Sequential()
        self.net.add_module("fc_in", nn.Linear(in_features=self.input_size, out_features=self.hid_size))
        self.net.add_module("act", nn.Tanh())
        self.rnn = klass(
            input_size=self.hid_size,
            hidden_size=self.hid_size,
            num_layers=self.rnn_layer,
            batch_first=True,
            dropout=self.dropout,
        )
        self.fc_out = nn.Linear(in_features=self.hid_size * 2, out_features=1)
        self.att_net = nn.Sequential()
        self.att_net.add_module(
            "att_fc_in",
            nn.Linear(in_features=self.hid_size, out_features=int(self.hid_size / 2)),
        )
        self.att_net.add_module("att_dropout", torch.nn.Dropout(self.dropout))
        self.att_net.add_module("att_act", nn.Tanh())
        self.att_net.add_module(
            "att_fc_out",
            nn.Linear(in_features=int(self.hid_size / 2), out_features=1, bias=False),
        )
        self.att_net.add_module("att_softmax", nn.Softmax(dim=1))

    def get_repre(self, inputs):
        # inputs: [batch_size, input_size*input_day]
        inputs = inputs.view(len(inputs), self.input_size, -1)
        inputs = inputs.permute(0, 2, 1)  # [batch, input_size, seq_len] -> [batch, seq_len, input_size]
        rnn_out, _ = self.rnn(self.net(inputs))  # [batch, seq_len, num_directions * hidden_size]
        attention_score = self.att_net(rnn_out)  # [batch, seq_len, 1]
        out_att = torch.mul(rnn_out, attention_score)
        out_att = torch.sum(out_att, dim=1)
        repres = torch.cat((rnn_out[:, -1, :], out_att), dim=1) # [batch, seq_len, num_directions * hidden_size]
        return repres
    
    def predict(self, repres):
        out = self.fc_out(
            repres
        )  # [batch, seq_len, num_directions * hidden_size] -> [batch, 1]
        return out[..., 0]
    
    def forward(self, inputs):
        outputs = self.get_repre(inputs)
        out = self.predict(outputs)
        return out


