import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW_FILE = "../../data/raw_data/clean_data.csv"
PROCESSED_DIR = Path("../../data/split_data")

df = pd.read_csv(RAW_FILE)

X = df.drop(columns=["silica_concentrate"])
y = df["silica_concentrate"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

