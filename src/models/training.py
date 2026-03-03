import pandas as pd
from pathlib import Path
import json
import pickle
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed_data"
SPLIT_DIR = BASE_DIR / "data" / "split_data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

X_train_scaled = pd.read_csv(PROCESSED_DIR / "X_train_scaled.csv")
y_train = pd.read_csv(SPLIT_DIR / "y_train.csv")


with open(BASE_DIR / "params" / "best_params.json", "r") as f:
    best_params = json.load(f)

model = RandomForestRegressor(
    **best_params,
    random_state=42
)

model.fit(X_train_scaled, y_train)

with open(MODEL_DIR / "model.pkl", "wb") as f:
    pickle.dump(model, f)

