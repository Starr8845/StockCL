# qlib 尝试检索出过去60天的x  以及过去60天以及未来5天的y

import qlib
from qlib.contrib.data.handler import Alpha360
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.handler import _DEFAULT_INFER_PROCESSORS, _DEFAULT_LEARN_PROCESSORS, check_transform_proc
from qlib.constant import REG_CN, REG_US
from qlib.tests.data import GetData
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord, RecordTemp
from qlib.utils import class_casting
from qlib.log import get_module_logger
from qlib.tests.config import CSI300_BENCH, CSI300_GBDT_TASK
import logging
import pandas as pd
import datetime
import numpy as np
from multi_y_alstm import ALSTM
from multi_y_gat import GAT
import argparse

def load_dataset(data_time, stock_set="csi300"):
    if data_time=="all":
        train_start_date = "2008-01-01"
        train_end_date = "2014-12-31"
        valid_start_date = "2015-01-01"
        valid_end_date = "2016-12-31"
        test_start_date = "2017-01-01"
        test_end_date = "2020-08-01"
    elif data_time=='toy2': 
        train_start_date = "2018-01-01"
        train_end_date = "2018-12-31"
        valid_start_date = "2019-01-01"
        valid_end_date = "2019-12-31"
        test_start_date = "2020-01-01"
        test_end_date = "2020-08-01"
    elif data_time=="new":
        train_start_date = "2008-01-01"
        train_end_date = "2017-12-31"
        valid_start_date = "2018-01-01"
        valid_end_date = "2019-12-31"
        test_start_date = "2020-01-01"
        test_end_date = "2023-08-01"
    elif data_time=="new2":
        train_start_date = "2008-01-01"
        train_end_date = "2016-12-31"
        valid_start_date = "2017-01-01"
        valid_end_date = "2018-12-31"
        test_start_date = "2019-01-01"
        test_end_date = "2022-08-01"
    elif data_time=="new3":
        train_start_date = "2014-07-30"
        train_end_date = "2021-7-30"
        valid_start_date = "2021-7-31"
        valid_end_date = "2022-7-31"
        test_start_date = "2022-08-01"
        test_end_date = "2023-08-01"
    elif data_time=="new4":
        train_start_date = "2014-01-01"
        train_end_date = "2018-12-31"
        valid_start_date = "2019-01-01"
        valid_end_date = "2019-12-31"
        test_start_date = "2020-01-01"
        test_end_date = "2020-12-31"
    # elif data_time=="new5":
    #     train_start_date = "2014-01-01"
    #     train_end_date = "2017-06-30"
    #     valid_start_date = "2017-07-01"
    #     valid_end_date = "2017-12-31"
    #     test_start_date = "2018-01-01"
    #     test_end_date = "2020-12-31"
    elif data_time=="new6":
        train_start_date = "2014-01-01"
        train_end_date = "2017-06-30"
        valid_start_date = "2017-07-01"
        valid_end_date = "2017-12-31"
        test_start_date = "2018-01-01"
        test_end_date = "2018-12-31"
    dh_config = {

        'start_time': datetime.datetime.strptime(train_start_date, '%Y-%m-%d'),

        'end_time': datetime.datetime.strptime(test_end_date, '%Y-%m-%d'),

        'fit_start_time': datetime.datetime.strptime(train_start_date, '%Y-%m-%d'),

        'fit_end_time': datetime.datetime.strptime(train_end_date, '%Y-%m-%d'),

        'infer_processors': [{'class': 'RobustZScoreNorm',

                            'kwargs': {'clip_outlier': True,

                                        'fields_group': 'feature'}},

                            {'class': 'Fillna',

                            'kwargs': {'fill_value':0,

                                'fields_group': 'feature'}}],

        'instruments': stock_set,

        'learn_processors': [
                            # {'class': 'DropnaLabel'},

                            {
                                'class': 'CSZScoreNorm',   
                                # 'class': 'CSRankNorm',
                            'kwargs': {'fields_group': 'label'}}],
        # 'learn_processors': [{'class': 'DropnaLabel'},],

    }
    handler = Alpha360_extend(**dh_config)
    dataset = DatasetH(handler=handler,segments={
        "train":[train_start_date, train_end_date],
        "valid":[valid_start_date, valid_end_date],
        "test":[test_start_date,test_end_date]
    })
    return dataset


class Alpha360_extend(DataHandlerLP):
    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=_DEFAULT_INFER_PROCESSORS,
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        filter_pipe=None,
        inst_processors=None,
        **kwargs
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            learn_processors=learn_processors,
            infer_processors=infer_processors,
            **kwargs
        )

    def get_label_config(self):
        fields = []
        labels = []
        # 把过去59天以及未来5天的数据都拿到
        for i in range(5,0,-1):
            fields.append(f"Ref($close, {i})/Ref($close, {i+1}) - 1")
            labels.append(f"LABEL{i}")
        fields.append(f"$close/Ref($close, 1) - 1")
        labels.append("LABEL0")
        fields.append(f"Ref($close, -1)/$close - 1")
        labels.append("LABEL-1")
        for i in range(-2,-6,-1):
            fields.append(f"Ref($close, {i})/Ref($close, {i+1}) - 1")
            labels.append(f"LABEL{i}")
        # print(fields)
        # print(labels)
        # return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]
        return fields, labels

    @staticmethod
    def get_feature_config():
        # NOTE:
        # Alpha360 tries to provide a dataset with original price data
        # the original price data includes the prices and volume in the last 60 days.
        # To make it easier to learn models from this dataset, all the prices and volume
        # are normalized by the latest price and volume data ( dividing by $close, $volume)
        # So the latest normalized $close will be 1 (with name CLOSE0), the latest normalized $volume will be 1 (with name VOLUME0)
        # If further normalization are executed (e.g. centralization),  CLOSE0 and VOLUME0 will be 0.
        fields = []
        names = []

        for i in range(59, 0, -1):
            fields += ["Ref($close, %d)/$close" % i]
            names += ["CLOSE%d" % i]
        fields += ["$close/$close"]
        names += ["CLOSE0"]
        for i in range(59, 0, -1):
            fields += ["Ref($open, %d)/$close" % i]
            names += ["OPEN%d" % i]
        fields += ["$open/$close"]
        names += ["OPEN0"]
        for i in range(59, 0, -1):
            fields += ["Ref($high, %d)/$close" % i]
            names += ["HIGH%d" % i]
        fields += ["$high/$close"]
        names += ["HIGH0"]
        for i in range(59, 0, -1):
            fields += ["Ref($low, %d)/$close" % i]
            names += ["LOW%d" % i]
        fields += ["$low/$close"]
        names += ["LOW0"]
        for i in range(59, 0, -1):
            fields += ["Ref($vwap, %d)/$close" % i]
            names += ["VWAP%d" % i]
        fields += ["$vwap/$close"]
        names += ["VWAP0"]
        for i in range(59, 0, -1):
            fields += ["Ref($volume, %d)/($volume+1e-12)" % i]
            names += ["VOLUME%d" % i]
        fields += ["$volume/($volume+1e-12)"]
        names += ["VOLUME0"]

        return fields, names


class SignalRecord3(RecordTemp):
    def __init__(self, recorder=None, data_region="cn"):
        super().__init__(recorder=recorder)
        self.data_region = data_region

    @staticmethod
    def generate_label(dataset, data_region):
        with class_casting(dataset, DatasetH):
            raw_label, = dataset.prepare(
                ["test"],
                col_set=["label"],
                data_key=DataHandlerLP.DK_R,
            )
            if data_region=="cn":
                raw_label=raw_label[('label','LABEL-2')].to_frame("label")  
            else:
                raw_label=raw_label[('label','LABEL-1')].to_frame("label")  
            raw_label = raw_label.fillna(0)
        return raw_label

    def save_label(self, dataset):
        raw_label = self.generate_label(dataset, self.data_region)
        # print("label中有无nan:", pred['score'].isnull().values.any())
        self.save(**{"label.pkl": raw_label})
    
    def save_pred(self, pred):
        if isinstance(pred, pd.Series):
            pred = pred.to_frame("score")
        self.save(**{"pred.pkl": pred})

    def generate(self, **kwargs):
        pass
            
    def list(self):
        return ["pred.pkl", "label.pkl"]


def get_parser():
    parser = argparse.ArgumentParser(description="commind parameter")
    parser.add_argument("-y_con", action="store_true", help="y_con")
    parser.add_argument("-memory", action="store_true", help="y_con")
    parser.add_argument("-data_time", dest="data_time", type=str, help="data_time", default="toy2")
    parser.add_argument("-base_model", dest="base_model", type=str, help="base_model", default="ALSTM")
    parser.add_argument("-stock_set", dest="stock_set", type=str, help="stock_set", default="csi300")
    parser.add_argument("-y_con_type", dest="y_con_type", type=str, help="y_con_type", default="proxy")
    parser.add_argument("-multi_predictor", action="store_true", help="multi_predictor")
    parser.add_argument("-M", dest="M", type=int, help="M", default=256)
    parser.add_argument("-threshold", dest="threshold", type=float, help="threshold", default=0.2)
    parser.add_argument("-ratio", dest="ratio", type=float, help="ratio", default=1)
    parser.add_argument("-save_path", dest="save_path", type=str, help="save_path", default="./models")
    parser.add_argument("-lam", dest="lam", type=float, help="lam", default=0.2)
    parser.add_argument("-all_negatives", action="store_true", help="all_negatives")
    args = parser.parse_args()
    return args


if __name__=="__main__":
    args = get_parser()
    logger = get_module_logger("run_workflow", logging.INFO)
    data_time=args.data_time
    y_con=args.y_con
    use_memory_net = args.memory
    base_model = args.base_model
    stock_set = args.stock_set
    if stock_set in ["csi300","csi500","csi1000"]:
        data_region="cn"
        if data_time=="all" and stock_set in ["csi300","csi500"]:
            provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
        else:
            provider_uri = '~/.qlib/qlib_data/crowd_data'
        GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
        qlib.init(provider_uri=provider_uri, region=REG_CN)
    else:
        data_region="usa"
        provider_uri = '~/.qlib/qlib_data/usa_data'
        GetData().qlib_data(target_dir=provider_uri, region=REG_US, exists_skip=True)
        qlib.init(provider_uri=provider_uri, region=REG_US)
    
    print(args.save_path)
    y_con_config={"y_con_type":"multi_y_contrast_loss","use_multi_y":False, "loss_weight":"all1", "lam":args.lam,"all_negative":args.all_negatives}
    # y_con_config = {"use_multi_y":False, "loss_weight":"all1", "y_con_type":"multi_y_contrast_loss"} 
    # y_con_config = {"y_con_type":"sampling","memory":use_memory_net, "y_con":y_con, "base_model":base_model}
    # y_con_config = {'y_con_type':"rankN"}
    # y_con_config = {'y_con_type':args.y_con_type}
    # y_con_config = {'y_con_type':'gaze'}
    # y_con_config = {'y_con_type':"feature_sim_con", "ratio":args.ratio}
    if base_model=="GATModel":
        model = GAT(n_epochs=100, GPU=0, seed=2023,     
                  use_memory_net=use_memory_net, use_y_model=False, use_sample_weighter=False, early_stop=15, 
                  base_model=base_model)
    else: # seed=2023
        import random
        seed = random.randint(1,100)
        print(f"seed={seed},prompt_num={args.M}, threshold={args.threshold}")
        model = ALSTM(n_epochs=100, GPU=0, seed=seed, batch_size=256,    # batch_size原来为256  # seed 原来为2023
                  use_memory_net=use_memory_net, use_y_model=False, use_sample_weighter=False, early_stop=15, 
                  base_model=base_model, data_region=data_region,
                  use_multi_predictor=args.multi_predictor,
                  prompt_num=args.M, threshold=args.threshold)
    recoder_name = f"{y_con}_{y_con_config}_{stock_set}_{data_time}"
    dataset = load_dataset(data_time=data_time, stock_set=stock_set)
    with R.start(experiment_name="contrastive_exp" if data_time!="toy2" else "toy", recorder_name=recoder_name):
        R.set_tags(data_time=data_time,base_model=base_model, y_con=y_con, use_memory_net=use_memory_net)
        recorder = R.get_recorder()
        # model.load_model("/home/zzx/quant/long_tail_qlib/temp/tcns/5.pt")

        sr = SignalRecord3(recorder=recorder,data_region=data_region)
        sr.save_label(dataset)
        if stock_set == "sp500":
            benchmark = "^gspc"
        elif stock_set == "nasdaq100":
            benchmark = "^ndx"
        elif stock_set == "csi300" or stock_set=="csi1000":
            benchmark = CSI300_BENCH
        elif stock_set=="csi500":
            benchmark = "SH000905"
        model.fit(dataset, recorder = recorder, save_path=args.save_path, y_con=y_con, y_con_config=y_con_config, sr=sr, benchmark=benchmark)
        R.save_objects(**{"params.pkl": model})
