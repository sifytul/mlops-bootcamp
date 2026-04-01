import pandas as pd
from sklearn.linear_model import LinearRegression
import os
import mlflow
import mlflow.sklearn

data = {
    "size": [500, 800, 1000, 1200, 1500, 1800],
    "price": [100000, 160000, 200000, 240000, 300000, 360000],
}

df = pd.DataFrame(data)

x = df[["size"]]
y = df["price"]

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

mlflow.autolog()  # Automatically log parameters, metrics, and models   
with mlflow.start_run():

    model = LinearRegression()
    model.fit(x, y)
    mlflow.sklearn.log_model(model, "model")

print("Model trained and logged in MLflow")