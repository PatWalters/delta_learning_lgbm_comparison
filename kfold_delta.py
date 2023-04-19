#!/usr/bin/env python

from sklearn.model_selection import KFold
import pandas as pd
from molfeat.calc import FPCalculator
from molfeat.trans import MoleculeTransformer
from lightgbm import LGBMRegressor
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from tqdm.auto import tqdm
import re
from glob import glob

def create_delta_df(df_in, activity_column ='Y'):
    merge_df = pd.merge(df_in,df_in,how='cross')
    merge_df['delta'] = merge_df[activity_column+"_x"] - merge_df[activity_column+"_y"]
    merge_df['combo_fp'] = [np.concatenate([a,b]) for a,b in merge_df[["fp_x","fp_y"]].values]
    return merge_df

def process_dataframe(df):
    cycle = 0
    result_list = []
    n_splits = 10
    n_jobs=-1
    for i in tqdm(range(0,5)):
        kf = KFold(n_splits=n_splits, random_state=i, shuffle=True)
        for train, test in tqdm(kf.split(df),total=n_splits):
            train = df.iloc[train].copy()
            test = df.iloc[test].copy()

            # build the traditional model
            lgbm_trad = LGBMRegressor(n_jobs=n_jobs,subsample=0.1, subsample_freq = 1)
            lgbm_trad.fit(np.stack(train.fp),train.Y)
            test['pred_trad'] = lgbm_trad.predict(np.stack(test.fp))
            # generate the delta dataframes
            train_delta_df = create_delta_df(train)
            test_delta_df = create_delta_df(test)
            # generate stats for the traditional model
            test_delta_df['pred_trad_delta'] = test_delta_df.pred_trad_x.values - test_delta_df.pred_trad_y.values
            r2_trad = r2_score(test_delta_df.delta,test_delta_df.pred_trad_delta)
            rmse_trad = mean_squared_error(test_delta_df.delta,test_delta_df.pred_trad_delta,squared=False)
            # build  the delta model
            lgbm_delta = LGBMRegressor(n_jobs=n_jobs,subsample=0.1, subsample_freq = 1)
            lgbm_delta.fit(np.stack(train_delta_df.combo_fp),train_delta_df.delta)
            delta_pred = lgbm_delta.predict(np.stack(test_delta_df.combo_fp))
            r2_delta = r2_score(test_delta_df.delta,delta_pred)
            rmse_delta = mean_squared_error(test_delta_df.delta,delta_pred,squared=False)

            result_list.append([cycle,r2_trad, r2_delta, rmse_trad, rmse_delta])
            cycle+=1
    return pd.DataFrame(result_list,columns=["cycle","r2_trad","r2_delta","rmse_trad","rmse_delta"])

def main():
    dataset_re = re.compile("Benchmarks/([^.]+)\.csv")
    calc = FPCalculator("ecfp")
    trans = MoleculeTransformer(calc)
    
    result_df_list = []
    for filename in sorted(glob("data/Benchmarks/*.csv")):
        dataset_name = dataset_re.findall(filename)[0]
        df = pd.read_csv(filename)
        if len(df) < 1300:
            print(dataset_name)
            df['fp'] = trans.transform(df.SMILES.values)
            result_df = process_dataframe(df)
            result_df['dataset'] = dataset_name
            result_df_list.append(result_df)

    pd.concat(result_df_list).to_csv("benchmark_data_subsamp.csv",index=False)
    

if __name__ == "__main__":
    main()
    




