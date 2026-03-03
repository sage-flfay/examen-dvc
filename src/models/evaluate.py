import pickle
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, root_mean_squared_error
import json

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed_data"
SPLIT_DIR = BASE_DIR / "data" / "split_data"
METRICS_DIR = BASE_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

with open(BASE_DIR / "models" / "model.pkl", "rb") as f:
    best_model = pickle.load(f)

X_test_scaled = pd.read_csv(PROCESSED_DIR / "X_test_scaled.csv")
y_test = pd.read_csv(SPLIT_DIR / "y_test.csv")


y_pred = best_model.predict(X_test_scaled)

rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

predictions_df = X_test_scaled.copy()
predictions_df["true_silica_concentrate"] = y_test.values
predictions_df["predicted_silica_concentrate"] = y_pred
predictions_df["residual"] = (
    predictions_df["true_silica_concentrate"] -
    predictions_df["predicted_silica_concentrate"]
)

predictions_df.to_csv(METRICS_DIR / "predictions.csv", index=False)

with open(METRICS_DIR / "scores.json", "w") as f:
    json.dump({"rmse": rmse, "r2": r2}, f, indent=4)
