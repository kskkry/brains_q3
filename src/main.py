import gc
import os
import pickle
import sys
import warnings
from multiprocessing import freeze_support

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.ensemble import *
from sklearn.preprocessing import *
from sklearn.preprocessing import StandardScaler, data

from config import Config, Stacking_MODEL
from make_dataset import MordredDataset, RdkitDataset
from model import LGBMMODEL
from mordred import Calculator, descriptors
from transform import TransformDataset

'''
実行
#$ cat sample/sample.in | python3 src/main.py
#python ./src/main.py < ./sample/sample.in
'''

warnings.simplefilter('ignore')
config = Config
stacking_model = Stacking_MODEL

ss = pickle.load(open('./src/scale_model/mlp_ss_scaler.pkl','rb'))

def input_dataset()-> pd.DataFrame:
    input_data = []
    for line in sys.stdin:
        input_data.append(str(line.strip()))
    input_df = pd.DataFrame(data=input_data, columns=["SMILES"])
    input_df["INDEX"] = 0
    return input_df 


def to_mordred(df: pd.DataFrame):
    """
    https://github.com/mordred-descriptor/mordred/blob/develop/examples/030-multiple_mol-multiple_desc.py
    サブミット時にはsrcフォルダ直下にmordredのソースコードを配置し直接インポートすることで対処した
    参考
    1. https://anaconda.org/mordred-descriptor/mordred/files
    2. https://pypi.org/project/mordred/
    """

    freeze_support()
    mols = [Chem.MolFromSmiles(smi) for smi in df["SMILES"]]
    calc = Calculator(descriptors, ignore_3D=True)
    mordred_df = calc.pandas(mols=mols, quiet=True)

    '''train時に全ての構造式に対してnullを示した特徴量を削除'''
    mordred_df = mordred_df.drop(config.mordred_ignore3D_True_all_null_feature_list, axis=1)

    mordred_df = mordred_df.astype(str).replace(" ", "")
    masks = mordred_df.apply(lambda d: d.str.contains('[a-zA-Z]' ,na=False))
    mordred_df[masks] = None
    all_data = pd.concat([df, mordred_df], axis=1)
    return all_data

def add_rdkit_dataset(data):
    rdf = RdkitDataset(data)
    data = rdf.transform()
    #print(data)
    return data

def add_mordred_dataset(data):
    mdd = MordredDataset(data)
    data = mdd.transform(train_data=False)
    return data


def transform(data):
    trf = TransformDataset(data)

    data = trf.transform()
    return data


def select(data):
    '''
    特徴量選択を行うメソッド。事前に得た特徴量の重要度(./src/similarity/feature_selection.csv)に基づいて
    ある閾値(config.FEATURE_SELECTION_THRESHOLD)以上の重要度をもつ特徴量のみを採用する
    '''
    feature_importance = pd.read_csv("./src/similarity/feature_selection.csv")
    list_feature_selected = list(feature_importance[feature_importance["feature_importance"] > config.FEATURE_SELECTION_THRESHOLD]["column"])
    data = data[list_feature_selected]
    return data

def main():
    all_data = input_dataset()
    all_data = to_mordred(all_data)
    all_data = transform(all_data) 
    all_data = all_data.drop(["SMILES", "INDEX"], axis=1)
    y_pred = np.ones((all_data.shape[0], ))
    
    MODEL_NUM = 1

    gbm_data = select(all_data)
    ss = pickle.load(open('./src/scale_model/mlp_ss_scaler.pkl','rb'))

    del all_data
    gc.collect()
    
    
    for model_name in stacking_model.lgb_models2:
        model = pickle.load(open(f'./src/models/{model_name}.pkl','rb'))
        y_pred *= model.predict(gbm_data.values, num_iteration=model.best_iteration_) 
    

    '''
    出力
    '''
    y_pred = np.power(y_pred, 1/8)
    y_pred = np.expm1(y_pred)
    for pred in y_pred:
        print(pred)

if __name__ == "__main__":
    main()

