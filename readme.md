# Dynamic Anomaly Detection Using API

## Project Overview
Dynamic Anomaly Detection Using API is a Python-based, end-to-end anomaly detection project designed to handle both small-scale and large-scale data. The system ingests data from APIs, preprocesses it, and automatically detects anomalies using multiple machine learning algorithms. The pipeline is config-driven, allowing users to switch between a Native path (for small datasets) and a Spark path (for large-scale datasets). Users can also provide input throughout the process, ensuring a human-in-the-loop approach for guiding anomaly detection decisions.

Key Features

  Config-Driven Pipeline: Users define processing mode and parameters via a configuration file, dynamically controlling the workflow.

  API Data Ingestion: Fetches and prepares data from any API endpoint for downstream processing.

  Dual Processing Paths:

    Native Path: Optimized for small-scale datasets.

    Spark Path: Optimized for large-scale datasets and distributed computation.

  Data Preprocessing: Cleans data, handles missing values, and scales features for ML models.

  Anomaly Detection:

    Native Path Algorithms: Isolation Forest, KMeans, DBSCAN with hyperparameter optimization.

    Spark Path Algorithms: KMeans, bisecting KMeans for large-scale clustering.

  Model Evaluation: Computes performance metrics to assess anomaly detection quality.

  Reporting & Visualization: Generates HTML reports and plots showing correlations, anomaly distributions, and model performance.

  User Interaction: Final results are presented for review, allowing users to accept, adjust, or rerun detection.

  Logging & Transparency: Tracks all processing steps to ensure reproducibility and auditability.

This architecture enables a flexible, scalable, and interactive anomaly detection system, suitable for both exploratory analysis and production deployment.
---
# Execution Model

This project is designed for on-demand, one-off anomaly detection analysis.
Each execution processes a snapshot of API data and produces a single consolidated HTML report containing data summary, model results, anomalies, visualizations, and logs.

The system emphasizes:

  Reproducibility: Each run records configuration, run metadata, and logs.

  Transparency: Every step from data ingestion to anomaly detection is logged.

  Interpretability: Users can review the full report, including anomaly counts, model scores, and visualizations.

  ⚠️ Note: This pipeline is not intended to run continuously, but for interactive snapshot-based analysis.

How to Use (Step-by-Step)

  Step 1 — Clone the project

    git clone https://github.com/MohVala/Dynamic_Anomaly_Detection_API
    cd dynamic-anomaly-detection


  Step 2 — Configure the pipeline

    Open config.yaml and specify:

      API endpoint(s)

      Processing mode: native (small datasets) or spark (large-scale datasets)

      Algorithm selection and hyperparameters

  Step 3 — Install dependencies

    pip install -r requirements.txt


  Step 4 — Run the analysis

    python main.py


    The pipeline dynamically follows either the Native path or the Spark path depending on the configuration.

    Steps include API ingestion → preprocessing → anomaly detection → evaluation → report generation.

  Step 5 — Review outputs

    The entire result is saved in a single HTML report under reports/:

    reports/
    └── report_.html


# Architecture

```text
                                ┌──────────────────────┐
                                │   User Configuration │
                                │   & Input Choices    │
                                └─────────┬────────────┘
                                          │
                                          ▼
                                ┌────────────────────┐
                                │    API Ingestion   |
                                │   (prepare data)   |
                                └─────────┬──────────┘
                                          │
                     ┌────────────────────┴────────────────────┐
                     │                                         │
                     ▼                                         ▼
        ┌────────────────────────┐                  ┌────────────────────────┐
        │       Native Path      │                  │        Spark Path      │
        │  (small-scale data)    │                  │   (large-scale data)   │
        └─────────┬──────────────┘                  └─────────┬──────────────┘
                  │                                         │
                  ▼                                         ▼
        ┌────────────────────────┐                  ┌────────────────────────┐
        │   Data Preprocessing   │                  │   Data Preprocessing   │
        │  (cleaning, scaling)   │                  │  (cleaning, scaling)   │
        └─────────┬──────────────┘                  └─────────┬──────────────┘
                  │                                         │
                  ▼                                         ▼
        ┌────────────────────────┐                  ┌────────────────────────┐
        │  Anomaly Detection     │                  │  Anomaly Detection     │
        │ (Isolation Forest,     │                  │ (  KMeans              │
        │  KMeans, DBSCAN)       │                  │  , bisecting_kmeans)   │
        │  + Hyperparameter      │                  │                        │
        │  Optimization          │                  │                        │
        └─────────┬──────────────┘                  └─────────┬──────────────┘
                  │                                         │
                  └───────────────┐     ┌───────────────────┘
                                  ▼     ▼
                        ┌────────────────────────┐
                        │ Model Evaluation &     │
                        │ Performance Metrics    │
                        └─────────┬──────────────┘
                                  │
                                  ▼
                        ┌──────────────────────────┐
                        │ Reporting & Visualization│
                        │ (HTML/Plots)             │
                        └─────────┬────────────────┘
                                  │
                                  ▼
                        ┌────────────────────────┐
                        │ User Decision &        │
                        │ Interaction            │
                        └────────────────────────┘

# Dynamic Anomaly Detection Using API

  An End-to-End Automated Pipeline for API-Based Anomaly Detection

  This project is a complete anomaly detection pipeline that ingests data from any REST API, cleans and preprocesses it, runs multiple anomaly detection algorithms, selects the best-performing model, and generates a consolidated HTML report with insights, visualizations, and logs.

  It demonstrates real-world skills in:

    API ingestion

    Data preprocessing

    Missing value imputation using ML

    Unsupervised anomaly detection

    Hyperparameter search

    Model evaluation & selection

    Dynamic reporting

    Logging & monitoring

    Scalable processing using Spark

  Key Features
  1. API Ingestion

    Accepts any REST API URL

    Automatically flattens JSON (handles nested objects/lists)

    Converts into Pandas DataFrame (or Spark DataFrame for large datasets)

    Performs basic type conversion and error handling

  2. Data Quality Processing

    Duplicate removal

    Missing value imputation using KMeans cluster-centroid method

    Feature normalization (MinMaxScaler)

    Provides summaries of:

      Shape

      Missing values

      Column types

      Duplicates

    Supports both Native (Pandas) and Spark (distributed) preprocessing paths depending on dataset size.

  3. Multiple Anomaly Detection Models

    Runs and compares different unsupervised models depending on the processing path:

    Processing Path	Algorithm	Purpose
    Native	Isolation Forest	Detect outliers using random isolation trees
      KMeans	Detect anomalies based on distance from cluster centers
      DBSCAN	Density-based anomaly detection
    Spark	KMeans	Scalable clustering for large datasets
      Bisecting KMeans	Efficient hierarchical clustering for big data

    Hyperparameter optimization is applied using ParameterGrid.

  4. Simple vs Complex Hyperparameter Modes

    Simple Mode: Fast runs with narrow hyperparameter ranges

    Complex Mode: Wider hyperparameter search for accuracy optimization

    This demonstrates understanding of runtime vs accuracy trade-offs.

  5. Model Evaluation & Selection

    Models are evaluated using Silhouette Score or similar metrics

    Best model is automatically selected

    Report includes:

      Anomaly counts and percentages

      Performance scores of all models

  6. Reporting & Visualization

    All outputs are consolidated in a single HTML report containing:

      Run metadata (Run ID, API URL, timestamp)

      Data summary (first rows, column types, missing values)

      Modeling results (scores and hyperparameters)

      Best model summary with anomaly count and percentage

    Visualizations:

      Correlation heatmaps (optional)

      Anomaly distributions

      Model performance comparisons

      Full logs of the pipeline execution

    Users can open the HTML report in a browser to review all results interactively.
    The report reflects either the Native path or Spark path, depending on configuration.