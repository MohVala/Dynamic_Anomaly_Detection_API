import os
import webbrowser
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64


def plot_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_base64


def native_generate_html_report(
    api_url, df, normed_df, result_dict, logs, output_dir="reports"
):
    os.makedirs(output_dir, exist_ok=True)
    html_file = os.path.join(output_dir, "report_.html")

    # 1. Data Summary Table
    data_summary_html = df.head().to_html(classes="table table-striped", index=False)
    dtype_html = df.dtypes.to_frame("dtype").to_html(classes="table table-striped")
    null_html = (
        df.isnull().sum().to_frame("nulls").to_html(classes="table table-striped")
    )

    # 2. modeling results table
    result_sum = []
    for method, item in result_dict.items():
        row = {
            "Model": item["algorithm"],
            "Score": item["score"],
            "Hyperparameters": item["parameters"],
        }
        result_sum.append(row)
    result_df = pd.DataFrame(result_sum).sort_values(by="Score", ascending=False)
    result_html = result_df.to_html(classes="table table-striped", index=False)

    # 3. Model Score bar chart
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.barplot(data=result_df, x="Model", y="Score", ax=ax1)
    plt.title("Model Scores (Sillhouette)")
    img_scores = plot_to_base64(fig1)

    # 4. Best Model anomaly chart
    best_model = max(result_dict, key=lambda x: result_dict[x]["score"])
    anomalies = pd.Series(result_dict[best_model]["anomaly_detection"])
    anomaly_counts = anomalies.value_counts().reindex([1, 0], fill_value=0)

    fig2, ax2 = plt.subplots(figsize=(4, 4))
    sns.barplot(
        x=anomaly_counts.index.map({1: "Anomaly", 0: "Normal"}),
        y=anomaly_counts.values,
        ax=ax2,
    )
    plt.title(f"Anomaly Counts ({best_model})")
    img_anomaly = plot_to_base64(fig2)

    # 5. Logs formatting
    logs_html = f"<pre>{logs}</pre>"

    # 6. HTML Content
    html_content = f"""
    <html>
    <head>
    <title>Anomaly Detection Report </title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/css/bootstrap.min.css">
    </head>
    <body class="p-4">
    <h1>Anomaly Detection Report</h1>
    <h3>Run ID: </h3>
    <h4>API URL: {api_url}</h4>
    <h4>Run Time: {datetime.now()}</h4>
    
    <hr>
    <h2>Data Summary</h2>
    <h5>First 5 Rows</h5>
    {data_summary_html}
    <h5>Column Types</h5>
    {dtype_html}
    <h5>Missing Values</h5>
    {null_html}
    
    <hr>
    <h2>Modeling Results</h2>
    {result_html}
    <img src="data:image/png;base64,{img_scores}" alt="Model Scores">
    
    <hr>
    <h2>Best Model: {best_model}</h2>
    <p>Score: {result_dict[best_model]['score']}</p>
    <p>Number of anomalies: {anomaly_counts[1]}</p>
    <p>Percentage of anomalies: {anomaly_counts[1]*100/anomaly_counts.sum():.2f}%</p>
    <img src="data:image/png;base64,{img_anomaly}" alt="Anomaly Counts">
    
    <hr>
    <h2>Logs</h2>
    {logs_html}
    
    </body>
    </html>
    """

    # Save HTML
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    #print(f" HTML report save to {html_file}")
