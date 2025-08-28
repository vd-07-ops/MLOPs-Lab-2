import pandas as pd
import numpy as np
import yaml
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load params
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

data = pd.read_csv(params["dataset"])

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Scaling
if params["preprocessing"]["scale"]:
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
else:
    X = X.values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=params["training"]["test_size"],
    random_state=params["training"]["random_state"]
)

train = pd.DataFrame(X_train)
train["Outcome"] = y_train.values
test = pd.DataFrame(X_test)
test["Outcome"] = y_test.values

os.makedirs("data/processed", exist_ok=True)
train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)
