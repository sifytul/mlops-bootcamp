import joblib

model = joblib.load("model/model.pkl")

def predict(size: float):
    prediction = model.predict([[size]])
    return prediction[0]