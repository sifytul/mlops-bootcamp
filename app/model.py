import mlflow.pyfunc
import mlflow


mlflow.set_tracking_uri("http://localhost:5000")
model_uri = "models:/mlops-model/1"
model = mlflow.pyfunc.load_model(model_uri)

def predict(size: float):
    prediction = model.predict([[size]])
    return prediction[0]