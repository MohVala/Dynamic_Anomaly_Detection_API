from typing import Dict, List, Optional, Any, Union, Tuple, Literal
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans


# duplicates handling:
def duplications(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(keep="first")


# Missing Value handling using ML modeling:


def fill_missing_with_kmeans(df: pd.DataFrame, n_cluster: int = 5) -> pd.DataFrame:
    df_filled = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols]

    df_train = df_numeric.dropna()

    if df_train.empty:
        print("❌ Not enough data to train KMeans (all rows have NaNs).")
        return df

    scaler = StandardScaler()
    scalerd_train = scaler.fit_transform(df_train)

    kmeans = KMeans(n_clusters=n_cluster, random_state=42)
    kmeans.fit(scalerd_train)

    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

    for idx in df.index:
        row = df_numeric.loc[idx]
        if row.isnull().any():
            temp_row = row.fillna(df_train.mean())
            temp_scaled = scaler.transform([temp_row])[0]
            cluster = kmeans.predict([temp_scaled])[0]
            cluster_center = cluster_centers[cluster]
            for col in row.index[row.isnull()]:
                df_filled.at[idx, col] = cluster_center[df_numeric.columns.get_loc(col)]

    return df_filled


# Normalization


def normalization_features(df: pd.DataFrame) -> pd.DataFrame:
    df_normalized = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numerics = df[numeric_cols]

    scaler = MinMaxScaler()
    df_normalized[df_numerics.columns] = scaler.fit_transform(df_numerics)

    return df_normalized
