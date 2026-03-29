import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

data = {
    "size": [500, 800, 1000, 1200, 1500, 1800],
    "price": [100000, 160000, 200000, 240000, 300000, 360000],
}

df = pd.DataFrame(data)

x = df[["size"]]
y = df["price"]

model = LinearRegression()
model.fit(x, y)

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")

print("Model trained and saved to model/model.pkl")