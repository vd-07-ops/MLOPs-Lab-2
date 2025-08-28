import pandas as pd
import yaml
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load parameters
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

train = pd.read_csv("data/processed/train.csv")

X_train = train.drop("Outcome", axis=1)
y_train = train["Outcome"]

model = RandomForestClassifier(
    n_estimators=params["model"]["params"]["n_estimators"],
    max_depth=params["model"]["params"]["max_depth"],
    random_state=params["model"]["params"]["random_state"]
)

model.fit(X_train, y_train)

# Save model
import os
os.makedirs("models", exist_ok=True)
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)
