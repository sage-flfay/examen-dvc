import pickle
import pandas as pd
from pathlib import Path
from sklearn.base import r2_score
from sklearn.metrics import root_mean_squared_error
import json

PROCESSED_DIR = Path("../../data/processed_data")

with open("../../models/model.pkl", "rb") as f:
    best_model = pickle.load(f)

X_test_scaled = pd.read_csv(PROCESSED_DIR / "X_test_scaled.csv")
y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv")


y_pred = best_model.predict(X_test_scaled)

rmse = root_mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

predictions_df = X_test_scaled.copy()
predictions_df["true_silica_concentrate"] = y_test.values
predictions_df["predicted_silica_concentrate"] = y_pred
predictions_df["residual"] = (
    predictions_df["true_silica_concentrate"] -
    predictions_df["predicted_silica_concentrate"]
)

predictions_df.to_csv("../../metrics/predictions.csv", index=False)

with open("../../metrics/scores.json", "w") as f:
    json.dump({"rmse": rmse, "r2": r2}, f, indent=4)
