import pandas as pd
import os
import pickle
import json
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Load test data
test = pd.read_csv("data/processed/test.csv")
X_test = test.drop("Outcome", axis=1)
y_test = test["Outcome"]

# Load model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

y_pred = model.predict(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')

# Save results
results = {
    "accuracy": acc,
    "f1_score": f1
}
os.makedirs("experiments", exist_ok=True)
with open("experiments/results.json", "w") as f:
    json.dump(results, f, indent=4)

# Save metrics.json (for dvc metrics)
metrics = {
    "accuracy": acc,
    "f1": f1
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Save plots/metrics.json (for dvc plots)
os.makedirs("plots", exist_ok=True)
plot_data = [
    {"index": i, "y_true": int(y_test.iloc[i]), "y_pred": int(y_pred[i])}
    for i in range(len(y_test))
]
with open("plots/metrics.json", "w") as f:
    json.dump(plot_data, f, indent=4)
