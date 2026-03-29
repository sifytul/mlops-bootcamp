from fastapi import FastAPI
from app.schema import HouseRequest
from app.model import predict

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome to the House Price Prediction API"}


@app.post("/predict")
def get_prediction(request: HouseRequest):
    price = predict(request.size)
    return {"predicted_price": price}