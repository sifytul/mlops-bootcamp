import os
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
import mlflow

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient(MLFLOW_TRACKING_URI)

latest_versions = client.search_model_versions("name='mlops-model'")
latest_version = max(latest_versions, key=lambda v: int(v.version))



client.set_registered_model_alias(
    name="mlops-model",
    version=latest_version.version,
    alias="production"
)

print(f"Model version {latest_version.version} promoted to production alias.")