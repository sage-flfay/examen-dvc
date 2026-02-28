import pandas as pd
from pathlib import Path
import json
import pickle
from sklearn.ensemble import RandomForestRegressor

PROCESSED_DIR = Path("../../data/processed")

X_train_scaled = pd.read_csv(PROCESSED_DIR / "X_train_scaled.csv")
y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv")

with open("metrics/best_params.json", "r") as f:
    best_params = json.load(f)

model = RandomForestRegressor(
    **best_params,
    random_state=42
)

model.fit(X_train_scaled, y_train)

with open("models/flotation_model.pkl", "wb") as f:
    pickle.dump(model, f)

