from turtle import st

import joblib
import pandas as pd

from feature_engineering import create_features

model = joblib.load("best_model.pkl")

df = pd.read_csv("new_campaign.csv")

df = create_features(df)

if "Revenue" in df.columns:
    df = df.drop(columns=["Revenue"])

predictions = model.predict(df)

output = df.copy()

output["Predicted_Revenue"] = predictions

output.to_csv(
    "predictions.csv",
    index=False
)

print("Predictions saved successfully!")
print(output.head())
