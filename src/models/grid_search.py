import pickle
import json
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed_data"
SPLIT_DIR = BASE_DIR / "data" / "split_data"
PARAM_DIR = BASE_DIR / "params"
PARAM_DIR.mkdir(parents=True, exist_ok=True)


X_train_scaled = pd.read_csv(PROCESSED_DIR / "X_train_scaled.csv")
y_train = pd.read_csv(SPLIT_DIR / "y_train.csv")

model = RandomForestRegressor(random_state=42)

param_grid = {
    "n_estimators": [100, 300, 500],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train_scaled, y_train)



with open(PARAM_DIR / "best_params.json", "w", encoding="utf-8") as f:
    json.dump(grid_search.best_params_, f, indent=4)



