import mlflow.pyfunc
import mlflow
from dotenv import load_dotenv
import os

load_dotenv()


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

model_uri = "models:/mlops-model@production"
model = mlflow.pyfunc.load_model(model_uri)

def predict(size: float):
    prediction = model.predict([[size]])
    return prediction[0]