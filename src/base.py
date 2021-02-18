import lightgbm as lgb
import numpy as np
import pandas as pd
import pickle
import os

class Base:
    '''
    学習モデルのための基底クラス
    結果として内容が盛りだくさんになってしまった
    '''
    def __init__(self, train_df, target, model_name, seed, is_cv=True) -> None:
        self.train_df = train_df
        self.target = target
        self.model_name = model_name
        self.SEED = seed
        self.IS_CV = is_cv
        pass

    def file_save(self, index=None):
        if self.IS_CV==True:
            save_name = str(self.model_name) + f"_cv_num_leaves{self.model.num_leaves}_seed{self.SEED}_index{index}"
        else:
            save_name = str(self.model_name) + "_seed{}".format(self.SEED)
        pickle.dump(self.model, open(os.path.dirname(os.path.dirname(__file__)) +  "\\src\\models\\{}.pkl".format(save_name), "wb"))

    def file_load(self, index=None):
        '''
        モデルを読み込むメソッド
        '''
        load_name = str(self.model_name)
        if self.IS_CV==True:
            load_name = load_name + f"_cv_num_leaves{self.model.num_leaves}_seed{self.SEED}_index{index}"
        else:
            load_name = load_name + "_seed{}".format(self.SEED)
        load_name += ".pkl"
        with open(os.path.dirname(os.path.dirname(__file__)) +  "/src/models/{}".format(load_name), "rb") as fp:
            self.model = pickle.load(fp)










