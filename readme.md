# Dynamic Anomaly Detection Using API

![Architecture Diagram](architecture_diagram.png)

## Project Overview
**Dynamic Anomaly Detection Using API** is a Python-based project that ingests data from any API, cleans and preprocesses it, and automatically detects anomalies using multiple machine learning algorithms. This service needs user's decision by some input questions and interactions.

Key features:
- Data ingestion from API endpoints
- Data cleaning and missing value handling using ML (KMeans)
- Feature normalization
- Multiple anomaly detection algorithms:
  - Isolation Forest
  - KMeans clustering
  - DBSCAN
- Hyperparameter optimization using `ParameterGrid`
- Visualization of correlations, model performance, and anomaly distribution
- Logging of processing steps for transparency

---

## Architecture

```text
           +-------------------+
           |   API Endpoint    |
           +--------+----------+
                    |
                    v
           +-------------------+
           |  Data Ingestion   |
           | (requests -> DF)  |
           +--------+----------+
                    |
                    v
           +-------------------+
           | Data Preprocessing|
           | - Missing values  |
           | - Normalization   |
           +--------+----------+
                    |
                    v
           +-------------------+
           |  Anomaly Detection|
           | - IsolationForest |
           | - KMeans          |
           | - DBSCAN          |
           +--------+----------+
                    |
                    v
           +-------------------+
           |  Results & Reports|
           | - Best model      |
           | - Score comparison|
           | - Visualization   |
           +-------------------+

# 🚀 Dynamic Anomaly Detection Using API  
**An End-to-End Automated Pipeline for API-Based Anomaly Detection**

This project is a complete anomaly detection pipeline designed to ingest **any REST API**, clean and preprocess the data, run multiple anomaly detection algorithms, automatically select the best-performing model, and generate final anomaly insights.

It is built as a portfolio project to demonstrate real-world skills in:
- API ingestion  
- Data preprocessing  
- Missing value imputation using ML  
- Unsupervised anomaly detection  
- Hyperparameter search  
- Model evaluation & selection  
- Dynamic reporting  
- Logging & monitoring  

---

## 🔥 Key Features

### **1. API Ingestion**
- Accepts any REST API URL  
- Automatic JSON flattening (handles nested objects/lists)  
- Converts into Pandas DataFrame  
- Basic type conversion & error handling  

---

### **2. Data Quality Processing**
- Duplicate removal  
- Missing value filling using **KMeans cluster-centroid imputation**  
- Normalization using **MinMaxScaler**  
- Summary of:
  - shape  
  - missing values  
  - column types  
  - duplicates  

---

### **3. Multiple Anomaly Detection Models**
Runs and compares 3 unsupervised models:

| Algorithm | Purpose |
|----------|---------|
| **Isolation Forest** | Detect outliers using random isolation trees |
| **KMeans** | Detect anomalies based on distance from cluster centers |
| **DBSCAN** | Density-based anomaly detection |

---

### **4. Simple vs Complex Hyperparameter Modes**
The user can choose:

- **Simple Mode** → fast, narrow hyperparameter ranges  
- **Complex Mode** → wide hyperparameter search grid  

This demonstrates understanding of runtime vs accuracy tradeoffs.

---

### **5. Model Evaluation & Selection**
Models are compared using:

- **Silhouette Score**

The best model is automatically selected.

---

### **6. Reporting & Visualization**
Includes:

- Comparing model scores  
- Anomaly vs normal  
- Correlation heatmap (optional visualization mode)  
- Head of dataset with anomaly flags  
- Complete process logs shown at the end  
