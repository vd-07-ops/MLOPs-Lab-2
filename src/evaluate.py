import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

test = pd.read_csv("data/processed/test.csv")
X_test = test.drop("Outcome", axis=1)
y_test = test["Outcome"]

# Load model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Save results
results = {
    "accuracy": acc,
    "f1_score": f1
}

import os
os.makedirs("experiments", exist_ok=True)
with open("experiments/results.json", "w") as f:
    json.dump(results, f, indent=4)
