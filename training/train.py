import pandas as pd
from sklearn.linear_model import LinearRegression
import os
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()

data = {
    "size": [500, 800, 1000, 1200, 1500, 1800],
    "price": [100000, 160000, 200000, 240000, 300000, 360000],
}

df = pd.DataFrame(data)

x = df[["size"]]
y = df["price"]

model = LinearRegression()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
print(f"Using MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

mlflow.autolog()  # Automatically log parameters, metrics, and models   
with mlflow.start_run():

    model.fit(x, y)
    mlflow.sklearn.log_model(model, "model")

    mlflow.register_model(
            "runs:/{}/model".format(mlflow.active_run().info.run_id),
            "mlops-model"
        )
    


print("Model trained and logged in MLflow")