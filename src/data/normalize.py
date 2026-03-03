import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

SPLIT_DIR = Path("../../data/split_data")
PROCESSED_DIR = Path("../../data/processed_data")

X_train = pd.read_csv(SPLIT_DIR / "X_train.csv")
X_test = pd.read_csv(SPLIT_DIR / "X_test.csv")

X_train = X_train.drop(columns=["date"])#useless
X_test = X_test.drop(columns=["date"])#useless

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

X_train_scaled_df.to_csv(PROCESSED_DIR / "X_train_scaled.csv", index=False)
X_test_scaled_df.to_csv(PROCESSED_DIR / "X_test_scaled.csv", index=False)


