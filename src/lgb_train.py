import gc
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import Config
from model import LGBMMODEL
from transform import TransformDataset

"""
conda install -c mordred-descriptor mordred
conda env create -f env.yaml
conda activate fujifilm-q3
"""

config = Config

def input_dataset():
    '''
    mordredを用いて拡張した特徴量
    '''
    all_data = pd.read_csv("./datasets/mordred_train_dataset.csv")
    all_data = all_data.replace("", None)
    all_data = all_data.drop(config.mordred_ignore3D_True_all_null_feature_list, axis=1)
    return all_data

def log1p_scale(data):
    path="./src/similarity/log1p_col.txt"
    with open(path, 'r') as f:
        log1p_col_list = f.readlines()
    
    col_list=[]
    for name in log1p_col_list:
        col=str(name).replace("\n","")
        col_list.append(col)
        data[col] = np.sign(data[col].astype(np.float32)) * np.log1p(np.abs(data[col].astype(np.float32)))

    return data

def transform(data):
    trf = TransformDataset(data)
    data = trf.transform()
    return data

def scale(data: pd.DataFrame):
    ss = StandardScaler()
    data = ss.fit_transform(data)
    return data


def cv_lgb_train(data):
    target = data["IC50 (nM)"]
    target = np.log1p(target)
    smiles = data["SMILES"]
    data = data.drop(["SMILES", "IC50 (nM)"], axis=1)

    #CV
    model = LGBMMODEL(train_df=data, target=target, model_name="lgb", seed=config.SEED, is_cv=True)

    if config.USING_ALL_DATA:
        oof_valid, oof_score = model.fit()
        model.lgb_visualize_importance()
    else:
        model.select()
        #model.train_df=model.train_df.values
        oof_valid, oof_score = model.fit()

    logger.info(f"is_cv={model.IS_CV}, n_splits={config.LGB_SPLITS}, using_all_data={config.USING_ALL_DATA}, num_leaves={model.model.num_leaves}, seed={config.SEED}, threshold={config.FEATURE_SELECTION_THRESHOLD}, OOF msle SCORE={oof_score}")
    np.save(f"./oof/lgb_n_splits{config.LGB_SPLITS}_num_leaves{model.model.num_leaves}_seed{config.SEED}_threshold{config.FEATURE_SELECTION_THRESHOLD}", oof_valid)
    return

def single_lgb_train(data):
    target = data["IC50 (nM)"]
    target = np.log1p(target)
    data = data.drop(["SMILES", "IC50 (nM)"], axis=1)

    #単一モデル
    model = LGBMMODEL(train_df=data, target=target, model_name="lgb", seed=config.SEED, is_cv=False)
    model.fit()

if __name__ == "__main__":
    all_data = input_dataset()
    all_data = transform(all_data)
    cv_lgb_train(all_data)
    #single_lgb_train(all_data)

