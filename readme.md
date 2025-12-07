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
