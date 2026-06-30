import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


def train_model(df):

    # Features and Target
    X = df.drop(columns=["Revenue"])
    y = df["Revenue"]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ),
        "XGBoost": XGBRegressor(
            random_state=42
        )
    }

    comparison = []

    best_model = None
    best_name = None
    best_r2 = -999
    best_pred = None

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        comparison.append([name, mae, rmse, r2])

        print(f"MAE : {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²  : {r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_name = name
            best_pred = y_pred

    comparison = pd.DataFrame(
        comparison,
        columns=["Model", "MAE", "RMSE", "R2"]
    )

    print("\nModel Comparison")
    print(comparison)

    print(f"\nBest Model : {best_name}")

    joblib.dump(best_model, "best_model.pkl")
    print("✅ best_model.pkl saved")

    print(X.columns.tolist())   # DEBUG

    joblib.dump(X.columns.tolist(), "feature_columns.pkl")
    print("✅ feature_columns.pkl saved")

    print("Model Saved Successfully!")
    

    return comparison, best_model, X, y, X_test, y_test, best_pred