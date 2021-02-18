from sklearn.metrics import mean_squared_log_error as msle
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from base import Base
from config import Config
from transform import TransformDataset
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import numpy as np
import pandas as pd
import pickle
import os

config = Config

class LGBMMODEL(Base):
    def __init__(self, train_df, target, model_name, seed, is_cv) -> None:
        super().__init__(train_df, target, model_name, seed, is_cv=is_cv)
        
        self.num_leaves=config.NUM_LEAVES
        self.models = []
        pass

    def setting(self):
        self.model = lgb.LGBMRegressor(
            boosting_type='gbdt', num_leaves=config.NUM_LEAVES, learning_rate=0.015, 
            n_estimators=10000, n_jobs=-1, random_state=self.SEED, 
            importance_type='gain', objective='rmse', max_bin=256
        )
        

        '''
        {'boosting_type': 'gbdt', 'objective': 'regression', 'metric': 'rmse', 'max_depth': 5, 'feature_pre_filter': False, 'lambda_l1': 7.800246235747819e-06, 'lambda_l2': 0.00010183724745272986, 'num_leaves': 8, 'feature_fraction': 0.45199999999999996, 'bagging_fraction': 1.0, 'bagging_freq': 0, 'min_child_samples': 25}
        {'boosting_type': 'gbdt', 'objective': 'regression', 'metric': 'rmse', 'max_depth': 5, 'feature_pre_filter': False, 'lambda_l1': 5.0539032107761205e-05, 'lambda_l2': 7.083914589957983e-07, 'num_leaves': 30, 'feature_fraction': 0.5, 'bagging_fraction': 1.0, 'bagging_freq': 0, 'min_child_samples': 20}
        '''

    def fit(self):
        if self.IS_CV==True:
            kf = KFold(n_splits=config.LGB_SPLITS, shuffle=True, random_state=config.SEED)
            oof_valid = np.zeros((self.train_df.shape[0],))
            '''
            np.log1p(target) mean=5.696034457842397, std=2.0097708208223373
            '''

            for index, (train_index, valid_index) in enumerate(kf.split(X=self.train_df)):
                print(f"{index+1}th/{config.LGB_SPLITS} Started")
                #X_train, y_train = self.train_df.values[train_index], self.target.values[train_index]
                #X_valid, y_valid = self.train_df.values[valid_index], self.target.values[valid_index]
                X_train, y_train = self.train_df[train_index], self.target[train_index]
                X_valid, y_valid = self.train_df[valid_index], self.target[valid_index]


                '''Q3 12/28までモデルを初期化してない期間があった(self.settingのような初期化を行っていなかった)'''
                self.setting()
                self.model.fit(
                    X=X_train, y=y_train, verbose=50,
                    eval_set=[(X_valid, y_valid)],
                    eval_metric=['rmse'], 
                    early_stopping_rounds=100
                )

                oof_pred = self.model.predict(X=X_valid, num_iteration=self.model.best_iteration_)

                oof_valid[valid_index] = np.expm1(oof_pred)
                oof_valid[oof_valid<0] = 0
                
                self.models.append(self.model)
                self.file_save(index=index+1)

            oof_score = msle(y_true=np.expm1(self.target), y_pred=oof_valid)
            print("oof score : {}".format(oof_score))
            oof_mse_score = mse(self.target, np.log1p(oof_valid))
            print("oof mse score : {}".format(oof_mse_score))
            sns.distplot(np.log1p(oof_valid), label="oof_valid")
            sns.distplot(self.target, label="target")
            plt.legend()
            plt.savefig("./img/oof_vs_target.png")
            plt.show()
            return oof_valid, oof_score
        else:
            self.setting()
            self.model.fit(
                X=self.train_df, y=self.target, verbose=50,
            )
            self.file_save(index=None)
    
    def predict(self, test_data):
        pred = self.model.predict(test_data) #単一モデルの場合のみ
        #pred = self.model.predict(test_data, num_iteration=self.model.best_iteration_)
        return pred

    def lgb_visualize_importance(self):
        """
        atma cupより
        lightGBM の model 配列の feature importance を plot する
        CVごとのブレを boxen plot として表現します.

        args:
            models:
                List of lightGBM models
            train_df:
                学習時に使った DataFrame
        """
        feature_importance_df = pd.DataFrame()
        for i, model in enumerate(self.models):
            _df = pd.DataFrame()
            _df['feature_importance'] = model.feature_importances_
            _df['column'] = self.train_df.columns
            _df['fold'] = i + 1
            feature_importance_df = pd.concat([feature_importance_df, _df], axis=0, ignore_index=True)

        order = feature_importance_df.groupby('column')\
            .sum()[['feature_importance']]\
            .sort_values('feature_importance', ascending=False).index[:50]

        fig, ax = plt.subplots(figsize=(max(6, len(order) * .4), 7))
        sns.boxenplot(data=feature_importance_df, x='column', y='feature_importance', order=order, ax=ax, palette='viridis')
        ax.tick_params(axis='x', rotation=90)
        ax.grid()
        fig.tight_layout()
        plt.show()
        fig.savefig('./img/lgb_importance.png')

        feature_selection_df = feature_importance_df.groupby('column')\
            .sum()[['feature_importance']]\
            .sort_values('feature_importance')
        feature_selection_df.to_csv("./src/similarity/feature_selection.csv")
        return fig, ax

    def select(self):
        '''
        特徴量選択を行うメソッド。事前に得た特徴量の重要度(./src/similarity/feature_selection.csv)に基づいて
        ある閾値(config.FEATURE_SELECTION_THRESHOLD)以上の重要度をもつ特徴量のみを採用する
        '''
        feature_importance = pd.read_csv("./src/similarity/feature_selection.csv")
        list_feature_selected = list(feature_importance[feature_importance["feature_importance"] > config.FEATURE_SELECTION_THRESHOLD]["column"])
        self.train_df = self.train_df[list_feature_selected]
        return