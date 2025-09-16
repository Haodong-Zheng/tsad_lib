'''
筛选离散和连续特征，返回离散和连续特征在原始数据中的列索引
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 筛选参数
DISCRETE_UNIQUE_THRESHOLD = 10
DISCRETE_RATIO_THRESHOLD = 0.05
CORR_THRESHOLD = 0.95

def filter():
    # 数据读取
    raw_df = pd.read_csv("data/sdc/metric.csv")
    df = raw_df.copy()
    timestamp_col = df.columns[0]
    original_columns = df.columns.tolist()  # 保存原始列名顺序
    df = df.drop(columns=[timestamp_col])
    
    # ===== 1. 去除标准差为 0 的字段 =====
    num_std = df.select_dtypes(include=[np.number]).std()
    zero_std_cols = num_std[num_std == 0].index.tolist()
    df.drop(columns=zero_std_cols, inplace=True)

    # ===== 2. 去除缺失率高的字段 =====
    missing_threshold = 0.9
    high_missing_cols = df.columns[df.isnull().mean() > missing_threshold].tolist()
    df.drop(columns=high_missing_cols, inplace=True)

    # ===== 3. 判断离散 vs 连续字段 =====
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

    # ===== 4. 删除每行唯一的离散字段 =====
    unique_discrete_cols = [
        col for col in discrete_cols
        if df[col].nunique() == len(df) and col != "timestamp"
    ]
    df.drop(columns=unique_discrete_cols, inplace=True)
    discrete_cols = [col for col in discrete_cols if col not in unique_discrete_cols]

    # ===== 5. 连续字段中去除高度相关特征 =====
    if continuous_cols:
        corr_matrix = df[continuous_cols].corr()
        groups = []
        visited = set()
        for col in continuous_cols:
            if col not in visited:
                group = set()
                stack = [col]
                while stack:
                    node = stack.pop()
                    if node not in visited:
                        visited.add(node)
                        group.add(node)
                        neighbors = corr_matrix.columns[
                            (corr_matrix[node].abs() > CORR_THRESHOLD) &
                            (corr_matrix.index != node)].tolist()
                        stack.extend([n for n in neighbors if n not in visited])
                groups.append(group)

        to_drop = []
        for group in groups:
            if len(group) > 1:
                avg_abs_corr = {}
                for feature in group:
                    other_features = [f for f in group if f != feature]
                    avg_abs_corr[feature] = corr_matrix.loc[feature, other_features].abs().mean()
                representative = max(avg_abs_corr, key=avg_abs_corr.get)
                to_drop.extend([f for f in group if f != representative])

        df.drop(columns=to_drop, inplace=True)
        continuous_cols = [col for col in continuous_cols if col not in to_drop]

    # ===== 6. 离散字段中去除高度相关特征 =====
    if discrete_cols:
        df_encoded = df[discrete_cols].copy()
        for col in df_encoded.columns:
            if df_encoded[col].dtype == 'object':
                df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

        if len(df_encoded.columns) > 1:
            corr_matrix_discrete = df_encoded.corr()
            groups = []
            visited = set()
            for col in discrete_cols:
                if col not in visited:
                    group = set()
                    stack = [col]
                    while stack:
                        node = stack.pop()
                        if node not in visited:
                            visited.add(node)
                            group.add(node)
                            neighbors = corr_matrix_discrete.columns[
                                (corr_matrix_discrete[node].abs() > CORR_THRESHOLD) &
                                (corr_matrix_discrete.index != node)].tolist()
                            stack.extend([n for n in neighbors if n not in visited])
                    groups.append(group)

            to_drop_discrete = []
            for group in groups:
                if len(group) > 1:
                    avg_abs_corr = {}
                    for feature in group:
                        other_features = [f for f in group if f != feature]
                        avg_abs_corr[feature] = corr_matrix_discrete.loc[feature, other_features].abs().mean()
                    representative = max(avg_abs_corr, key=avg_abs_corr.get)
                    to_drop_discrete.extend([f for f in group if f != representative])

            df.drop(columns=to_drop_discrete, inplace=True)
            discrete_cols = [col for col in discrete_cols if col not in to_drop_discrete]

    # ===== 转换列名为原始索引 =====
    # 获取最终保留的所有列名
    final_columns = discrete_cols + continuous_cols
    
    # 获取这些列在原始数据中的索引位置（排除timestamp列）
    discrete_indices = [original_columns.index(col)-1 for col in discrete_cols]  # -1因为去掉了timestamp列
    continuous_indices = [original_columns.index(col)-1 for col in continuous_cols]
    
    # 索引按升序排列
    discrete_indices.sort()
    continuous_indices.sort()

    # print("discrete_indices:", discrete_indices)
    # print("continuous_indices:", continuous_indices)

    return discrete_indices, continuous_indices
