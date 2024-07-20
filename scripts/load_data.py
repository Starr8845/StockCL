import qlib
from qlib.contrib.data.handler import Alpha360
from qlib.data.dataset import DatasetH
import datetime


def load_dataset(data_time):
    if data_time=="all":
        train_start_date = "2008-01-01"
        train_end_date = "2014-12-31"
        valid_start_date = "2015-01-01"
        valid_end_date = "2016-12-31"
        test_start_date = "2017-01-01"
        test_end_date = "2020-08-01"
    elif data_time=="toy1":
        train_start_date = "2014-01-01"
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

        'instruments': 'csi300',

        'label': ['Ref($close, -2) / Ref($close, -1) - 1'],

        'learn_processors': [{'class': 'DropnaLabel'},

                            {
                                'class': 'CSZScoreNorm',   
                                # 'class': 'CSRankNorm',
                                'kwargs': {'fields_group': 'label'}
                            }
                            ],
        # 'learn_processors': [{'class': 'DropnaLabel'},],

    }
    handler = Alpha360(**dh_config)
    dataset = DatasetH(handler=handler,segments={
        "train":[train_start_date, train_end_date],
        "valid":[valid_start_date, valid_end_date],
        "test":[test_start_date,test_end_date]
    })
    return dataset


    