'''
识别离散和连续特征
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from configs import Config

# 参数
DISCRETE_UNIQUE_THRESHOLD = 10
DISCRETE_RATIO_THRESHOLD = 0.05

def reco():
    # 数据读取
    if Config.dataset == "SDC":
        raw_df = pd.read_csv("data/sdc/metric.csv")
    elif Config.dataset == "SMD":
        raw_df = pd.read_csv("data/SMD/machine-1-1_train.csv")
    # print(f"原始字段数: {raw_df.shape[1]}, 样本数: {raw_df.shape[0]}")
    df = raw_df.copy()
    # ===== 1. 去除第一列的时间戳 =====
    timestamp_col = df.columns[0]
    df = df.drop(columns=[timestamp_col])
    # print(df.shape)
    # print(f"[1] 已移除时间戳列: {timestamp_col}")

    # ===== 2. 判断离散 vs 连续字段 =====
    discrete_cols, continuous_cols = [], []
    for col in df.columns:
        series = df[col]
        if series.dtype == 'object':
            discrete_cols.append(col)
        elif np.issubdtype(series.dtype, np.number):
            nunique = series.nunique()
            ratio = nunique / len(series)
            if nunique < DISCRETE_UNIQUE_THRESHOLD or ratio < DISCRETE_RATIO_THRESHOLD:
                discrete_cols.append(col)
            else:
                continuous_cols.append(col)

    # print(f"[2] 识别离散字段数: {len(discrete_cols)}, 连续字段数: {len(continuous_cols)}")

    # ===== 3. 获取特征索引 =====
    discrete_idx = [df.columns.get_loc(col) for col in discrete_cols]
    continuous_idx = [df.columns.get_loc(col) for col in continuous_cols]

    # print("\n====== 处理完成 ======")
    # print(f"离散特征索引: {discrete_idx}")
    # print(f"连续特征索引: {continuous_idx}")
    return discrete_idx, continuous_idx
