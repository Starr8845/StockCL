import qlib
from qlib.constant import REG_CN
from qlib.tests.data import GetData
from alstm import ALSTM
# from load_data import load_dataset
from multi_y_run import load_dataset
import os
from base_models import MemoryModule
import torch as torch
import numpy as np

if __name__ == "__main__":
    # use default data
    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    data_time="all"
    dataset = load_dataset(data_time=data_time,stock_set="csi300")
    
    path = "/home/zzx/quant/long_tail_qlib/models/2023-12-14-ALSTM_mem"
    for i in range(17):
        print(i)
        print(f"{path}/repres_{i}")
        if not os.path.exists(f"{path}/repres_{i}"):
            print(f"{path}/repres_{i}")
            os.mkdir(f"{path}/repres_{i}")
        model = ALSTM(n_epochs=100, GPU=0, seed=2023, batch_size=2048,    # batch_size原来为256
                  use_y_model=False, use_sample_weighter=False, early_stop=15, 
                  base_model="ALSTM", data_region="cn")
        model.load_model(f'{path}/ALSTM_{i}.pt')
    
        model.get_sampled_repre(dataset, path=f"{path}/repres_{i}")

        memory_net = MemoryModule(hidden_size=128,prompt_num=256)

        memory_net.load_state_dict(torch.load(f"{path}/memory_{i}.pt"))

        print(memory_net.memory.detach())

        np.save(f"{path}/repres_{i}/memory_weight.npy",memory_net.memory.detach().cpu().numpy())


