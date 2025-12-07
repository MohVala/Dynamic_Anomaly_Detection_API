import logging
from io import StringIO


# -------------------------------
# Setup in-memory log with extra fields
# -------------------------------

log_stream = StringIO()
base_logger = logging.getLogger("anomaly_logger")
base_logger.setLevel(logging.INFO)

if base_logger.hasHandlers():
    base_logger.handlers.clear()

stream_handler = logging.StreamHandler(log_stream)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(step)s - %(stage)s - %(message)s')
stream_handler.setFormatter(formatter)
base_logger.addHandler(stream_handler)

def log(step, stage, message):
    extra = {"step": step, "stage": stage}
    base_logger.info(message, extra=extra)

# -------------------------------
# Import Libraries
# -------------------------------
log("import", "start", "Importing necessary libraries")
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import uuid
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark import SparkConf
import time
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
import optuna
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.special import expit
import multiprocessing
import seaborn as sns

log("import", "end", "Importing necessary libraries")


# ----------------------------
# Data Ingestion:
# ----------------------------
log("data_ingestion", "start", "data ingestion from source")

def ingest_api_to_dataframe(api_url):
    def flatten_json(y):
        out = {}

        def flatten(x, name=''):
            if isinstance(x, dict):
                for a in x:
                    flatten(x[a], f'{name}{a}_')
            elif isinstance(x,list):
                for i, a in enumerate(x):
                    flatten(a,f'{name}{i}_')
            else:
                out[name[:-1]] = x
            
        flatten(y)
        return out
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        raw_json = response.json()

        # handle list or single object:
        if isinstance(raw_json, list):
            data = [flatten_json(item) for item in raw_json]
        else:
            data = [flatten_json(raw_json)]
        return pd.DataFrame(data)
    
    except Exception as e:
        print(f" Error Happened ******** {e}***********")
        return pd.DataFrame()
    
API_URL = 'https://data.cityofnewyork.us/resource/biws-g3hs.json?$limit=10000' 
print(" Make sure all Data Types are correct. \n ****  This service does not soppurt to change data types.")

log("data_ingestion", "start", "data ingestion from source")
#input("Enter the API URL: ")
ingested_df = ingest_api_to_dataframe(API_URL)
ingested_df.head()   
log("data_ingestion", "end", "data ingestion from source")


print( " you have answer some questions before start: \n ***************** \n")

visualization_question = input("1. do you need visualization while runing? \n process will stop until you close visualization windows. \n asnwer y or n: \n")
model_complexity_question = input("2. do want to run more complicated and wider range of hyperparameters or simpler and shorter range? \n asnwer c: comlicated s: simple \n")
# Just for test:
numeric_cols = [
    'passenger_count', 'trip_distance', 'ratecodeid', 'pulocationid', 'dolocationid',
    'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount',
    'improvement_surcharge', 'total_amount'
]

for col in numeric_cols:
    ingested_df[col] = pd.to_numeric(ingested_df[col], errors='coerce')  # converts and sets invalid parsing to NaN
   
# ----------------------------
# Data Preparation:
# ----------------------------   
log("data_preparation", "start", "data preparation")
def summarize_dataframe(df):
    print("🔍 Data Summary:")
    print(f"- Shape: {df.shape}")
    print("\n📋 Column Types:")
    print(df.dtypes)

    print("\n🕳️ Missing Values per Column:")
    print(df.isnull().sum())

    print(f"\n🔁 Duplicate Rows: {df.duplicated().sum()}")
summarize_dataframe(ingested_df)
print(ingested_df.head())

# some usefull informations:
duplicates_count = ingested_df.duplicated().sum()
null_count = ingested_df.isnull().sum()

# remove duplicates and keep fisrt one only:
ingested_df.drop_duplicates(keep = 'first', inplace = True)

# Missing Value handling using ML modeling:

def fill_missing_with_kmeans(df, n_cluster = 5):
    df_filled = df.copy()

    numeric_cols = df.select_dtypes(include = [np.number]).columns
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

filled_df = fill_missing_with_kmeans(ingested_df, n_cluster=5)
filled_df['store_and_fwd_flag'].value_counts()
    
if visualization_question == 'y':
    # Features Correlation  
    df_cor = filled_df.select_dtypes(include=[np.number])
    corr = df_cor.corr()

    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()

# Normalization

def normalization_features(df):
    df_normalized = df.copy()

    numeric_cols = df.select_dtypes(include = [np.number]).columns
    df_numerics = df[numeric_cols]

    scaler = MinMaxScaler()
    df_normalized[df_numerics.columns] = scaler.fit_transform(df_numerics)

    return df_normalized

normed_df = normalization_features(filled_df)
print(normed_df)
log("data_preparation", "end", "data preparation")

# ----------------------------
# Modeling:
# ---------------------------- 

def get_time_series_split (X, n_splits = 5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return [(train_index, test_index) for train_index, test_index in tscv.split(X)]

# this function run all algorithms with different range of hyperparameters and find the best types of algorithms seperately:
if model_complexity_question == 's': # answered simple modeling  
    def run_all_anomaly_detectors(df, methods=['isolation_forest','kmeans','dbscan'], true_anomalies = None, save_models = False):
        result = {}
        data = df.loc[:, normed_df.dtypes !='object']
        for method in methods:
            if method == 'isolation_forest':
                log("simple_modeling_1/3", "start", "simple isolation forest modeling ")
                print(f"start of modeling: {method}")
                param_grid = {
                    'n_estimators': range(10,20, 10),
                    'max_samples': np.arange(0.1,0.4, 0.1),
                    'contamination': np.linspace(0.01, 0.5, 2),
                    'max_features': np.arange(0.1, 0.3, 0.1)
                }

                best_score = -1
                best_model = None
                best_params = None
                best_preds = None

                for params in ParameterGrid(param_grid):
                    model = IsolationForest(**params, random_state=42)
                    preds = model.fit_predict(data)
                    score = silhouette_score(data, preds)
                    if score>best_score:
                        best_score = score
                        best_model = model
                        best_params = params
                        best_preds = np.where(preds==-1,1,0)
                result[method] = {
                    'algorithm': "Isolation Forest",
                    'model' : best_model,
                    'score' : best_score,
                    'parameters': best_params,
                    'anomaly_detection': best_preds
                }
                print(f"end of modeling: {method} \n ====== \n")
                log("simple_modeling_1/3", "end", "simple isolation forest modeling ")


            elif method == 'kmeans':
                log("simple_modeling_2/3", "start", "simple kmeans modeling ")
                print(f"start of modeling: {method}")

                param_grid = {
                    'n_clusters': range(3,5, 1)
                }

                best_score = -1
                best_model = None
                best_params = None
                best_preds = None

                for params in ParameterGrid(param_grid):
                    model = KMeans(**params, random_state=42)
                    preds = model.fit_predict(data)
                    score = silhouette_score(data, preds)
                    
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_params = params
                        best_preds = preds
                result[method] = {
                    'algorithm': "KMeans",
                    'model' : best_model,
                    'score' : best_score,
                    'parameters': best_params,
                    'anomaly_detection': best_preds
                }

                print(f"end of modeling: {method} \n ====== \n")
                log("simple_modeling_2/3", "end", "simple kmeans modeling ")
            elif method == 'dbscan':
                log("simple_modeling_3/3", "start", "simple dbscan modeling ")
                print(f"start of modeling: {method}")

                param_grid = {
                    'eps' : np.arange(0.1,0.2, 0.1),
                    'min_samples': range(3, 7, 2)
                }

                best_score = -1
                best_model = None
                best_params = None
                best_preds = None

                for params in ParameterGrid(param_grid):
                    model = DBSCAN(**params)
                    preds = model.fit_predict(data)
                    score = silhouette_score(data, preds)
                    
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_params = params
                        best_preds = preds
                result[method] = {
                    'algorithm': "DBSCAN",
                    'model' : best_model,
                    'score' : best_score,
                    'parameters': best_params,
                    'anomaly_detection': best_preds
                }
                print(f"end of modeling: {method} \n ====== \n")
                log("simple_modeling_3/3", "end", "simple dbscan modeling ")
        return result


    result_dict = run_all_anomaly_detectors(normed_df.loc[:,normed_df.dtypes  != 'object'], methods=["isolation_forest", "kmeans", "dbscan"])

    for method in result_dict:
        print(f"{method}: {result_dict[method]['score']}")
else: # asnwered comlicated modeling
    def run_all_anomaly_detectors(df, methods=['isolation_forest','kmeans','dbscan'], true_anomalies = None, save_models = False):
        result = {}
        data = df.loc[:, normed_df.dtypes !='object']
        for method in methods:
            log("simple_modeling_1/3", "start", "complex isolation forest modeling ")
            if method == 'isolation_forest':
                print(f"start of modeling: {method}")
                param_grid = {
                    'n_estimators': range(10,100, 10),
                    'max_samples': np.arange(0.1,1, 0.1),
                    'contamination': np.linspace(0.01, 0.5, 10),
                    'max_features': np.arange(0.1, 1, 0.1)
                }

                best_score = -1
                best_model = None
                best_params = None
                best_preds = None

                for params in ParameterGrid(param_grid):
                    model = IsolationForest(**params, random_state=42)
                    preds = model.fit_predict(data)
                    score = silhouette_score(data, preds)
                    if score>best_score:
                        best_score = score
                        best_model = model
                        best_params = params
                        best_preds = np.where(preds==-1,1,0)
                result[method] = {
                    'algorithm': "Isolation Forest",
                    'model' : best_model,
                    'score' : best_score,
                    'parameters': best_params,
                    'anomaly_detection': best_preds
                }
                print(f"end of modeling: {method} \n ====== \n")
                log("simple_modeling_1/3", "end", "complex isolation forest modeling ")



            elif method == 'kmeans':
                log("simple_modeling_2/3", "start", "complex kmeans modeling ")

                print(f"start of modeling: {method}")

                param_grid = {
                    'n_clusters': range(3,10, 1)
                }

                best_score = -1
                best_model = None
                best_params = None
                best_preds = None

                for params in ParameterGrid(param_grid):
                    model = KMeans(**params, random_state=42)
                    preds = model.fit_predict(data)
                    score = silhouette_score(data, preds)
                    
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_params = params
                        best_preds = preds
                result[method] = {
                    'algorithm': "KMeans",
                    'model' : best_model,
                    'score' : best_score,
                    'parameters': best_params,
                    'anomaly_detection': best_preds
                }

                print(f"end of modeling: {method} \n ====== \n")
                log("simple_modeling_2/3", "end", "complex kmeans modeling ")
            elif method == 'dbscan':
                log("simple_modeling_3/3", "start", "complex dbscan modeling ")
                print(f"start of modeling: {method}")

                param_grid = {
                    'eps' : np.arange(0.1,1, 0.1),
                    'min_samples': range(3, 21, 2)
                }

                best_score = -1
                best_model = None
                best_params = None
                best_preds = None

                for params in ParameterGrid(param_grid):
                    model = DBSCAN(**params)
                    preds = model.fit_predict(data)
                    score = silhouette_score(data, preds)
                    
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_params = params
                        best_preds = preds
                result[method] = {
                    'algorithm': "DBSCAN",
                    'model' : best_model,
                    'score' : best_score,
                    'parameters': best_params,
                    'anomaly_detection': best_preds
                }
                print(f"end of modeling: {method} \n ====== \n")
                log("simple_modeling_3/3", "end", "complex dbscan modeling ")
        return result


    result_dict = run_all_anomaly_detectors(normed_df.loc[:,normed_df.dtypes  != 'object'], methods=["isolation_forest", "kmeans", "dbscan"])

    for method in result_dict:
        print(f"{method}: {result_dict[method]['score']}")


# explain about input data and missing values, duplicated
print(f"There {duplicates_count} duplicated records, which dropped and just kept first ones.")

null_count = ingested_df.isnull().sum()
print(f"Here you can see null values for every feature: \n {null_count} \n Null values deleted if existed.")
# explain best model and its score
best_model = max(result_dict, key = lambda x: result_dict[x]['score'])
best_score = result_dict[best_model]['score']
print(f"Best model is {best_model} with this score: {best_score}")

# compare all best models of different algorithms in one table and in some charts as well.
result_sum = []
for method, item in result_dict.items():
    row = {
        "Model": item['algorithm'],
        "Score": item['score'],
        "Hyperparameters": item['parameters']
        }
    result_sum.append(row)

result_sum = pd.DataFrame(result_sum)
result_sum.sort_values(by='score', ascending=False).head(10)
#compare result of models in chart
sns.barplot(data=result_sum, x='Model', y = 'score')
plt.title("Models Scores (Silhouette)")
plt.show()

# explain count and share of anomalies in best model
detected_anomaly = int(pd.Series(result_dict[best_model]['anomaly_detection']).value_counts()[1])
normal_data = int(pd.Series(result_dict[best_model]['anomaly_detection']).value_counts()[0])
percent_anomaly = detected_anomaly*100/(detected_anomaly+normal_data)
print(f"There are {detected_anomaly} detected anomalies in your data. \n it means %{percent_anomaly} of your data is detected as anomaly.")
sns.barplot(data=pd.DataFrame({
    "label": ["Anomaly", "Normal"],
    "value": [detected_anomaly, normal_data]}),
    x="label",
    y = "value")
plt.title("Count of Anomalies and Normal Data")
plt.show()
# explain about rows (records) of dataset which detected as anomaly
normed_df['anomaly_flag'] = result_dict[best_model]['anomaly_detection']
normed_df.head(20)


# print log of process
logs = log_stream.getvalue()
print(logs)
