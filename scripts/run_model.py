import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData
from qlib.tests.config import CSI300_BENCH, CSI300_GBDT_TASK
from qlib.contrib.data.handler import Alpha360
from qlib.data.dataset import DatasetH
import datetime
from alstm import ALSTM
from load_data import load_dataset
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="commind parameter")
    parser.add_argument("-temp", dest="temp", type=float, help="tempForContrast", default=0.1)
    parser.add_argument("-y_con", action="store_true", help="y_con")
    parser.add_argument("-data_time", dest="data_time", type=str, help="data_time", default="toy2")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # use default data
    args = get_parser()
    print(args)
    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    
    model = ALSTM(n_epochs=100, GPU=0, seed=2023, batch_size=2000)
    y_con_config = {"temp":args.temp, "threshold1":0.1, "threshold2":0.3, "threshold_cluster":0, "use_loss_weight":True} # 默认 threshold2为0.3
    dataset = load_dataset(data_time=args.data_time)

    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": (model, dataset),
                "topk": 50,
                "n_drop": 5,
            },
        },
        "backtest": {
            "start_time": "2017-01-01",
            "end_time": "2020-08-01",
            "account": 100000000,
            "benchmark": CSI300_BENCH,
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }

    with R.start(experiment_name="ALSTM"):
        # R.log_params(ders="temp...")
        R.set_tags(data_time=args.data_time, y_con=args.y_con, temp=args.temp)
        recorder = R.get_recorder()
        model.fit(dataset, recorder = recorder, save_path='./models/ALSTM.pt', y_con=args.y_con, y_con_config=y_con_config)
        # model.load_model("models/ALSTM_withoutrank_y_contrast_weight.pt")
        # model.get_sampled_repre(dataset=dataset, name="withoutrank_y_contrast_weight_raw_label")
        R.save_objects(**{"params.pkl": model})

        # prediction
        
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # Signal Analysis
        sar = SigAnaRecord(recorder)
        sar.generate()

        # backtest. If users want to use backtest based on their own prediction,
        # please refer to https://qlib.readthedocs.io/en/latest/component/recorder.html#record-template.
        # par = PortAnaRecord(recorder, port_analysis_config, "day")
        # par.generate()


