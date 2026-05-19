import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
)

# loading the data
data = load_breast_cancer()
x = data.data
y = data.target
df = pd.DataFrame(x, columns=data.feature_names)

# exploring the dataset
print("The first 5 samples:")
print(f"\n{df.head()}")

print("\n" + 50 * "=")

print(f"\nDataset shape: {x.shape}")
print(f"Features: {data.feature_names}")
print(f"\nTarget names: {data.target_names}")
print(f"\nNull values per column:\n{df.isnull().sum()}")
print(f"Duplicate rows:\n{df.duplicated().sum()}")

print("\nClass distribution:")
print(f"\nMalignan (0): {sum(y==0)} samples.")
print(f"\nBenign (1): {sum(y==1)} samples.")

# tuning knn
print("\n" + 50 * "=" + "\n")

x_temp, x_test_temp, y_temp, y_test_temp = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)
scaler_temp = StandardScaler()
x_temp_scaled = scaler_temp.fit_transform(x_temp)
x_test_temp_scaled = scaler_temp.fit_transform(x_test_temp)

k_values = range(1, 21)
k_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_temp_scaled, y_temp)
    y_pred = knn.predict(x_test_temp_scaled)
    k_scores.append(accuracy_score(y_test_temp, y_pred))
    print(f"k = {k:2d}: Accuracy = {k_scores[-1]:.4f}")

best_k = k_values[np.argmax(k_scores)]
print(f"\nBest k: {best_k} (Accuracy = {max(k_scores):.4f})")

plt.figure(figsize=(8, 5))
plt.plot(k_values, k_scores, "bo-", linewidth=2)
plt.xlabel("k value")
plt.ylabel("Accuracy")
plt.title("Finding the best k for KNN")
plt.grid(True, alpha=0.3)
plt.show()

# spliting data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

print("\n" + 50 * "=")

print("\nAfter the split:")
print(f"Training set: {x_train.shape[0]}")
print(f"Test set: {x_test.shape[0]}")

# preprocessing
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# training models
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=best_k),
    "SVC": SVC(random_state=42, probability=True),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
}
results = {}
roc_data = {}

for name, model in models.items():
    model.fit(x_train, y_train)  # trains the model

    y_pred = model.predict(x_test)  # makes predictions

    y_prob = model.predict_proba(x_test)[:, 1]

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "f1-score": f1_score(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, y_prob),
    }

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data[name] = (fpr, tpr, results[name]["AUC-ROC"])

# displaying results
print("\n" + 50 * "=")

print("\nModel performance")

results_df = pd.DataFrame(results).T
print(results_df.round(4))

# plotting
plt.figure(figsize=(8, 6))
colors = {
    "Logistic Regression": "blue",
    "KNN": "green",
    "SVC": "red",
    "Decision Tree": "orange",
}

for name, (fpr, tpr, auc) in roc_data.items():
    plt.plot(
        fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=colors[name], linewidth=2
    )


plt.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.5)")

plt.xlabel("False Positive rate")
plt.ylabel("True Positive rate")
plt.title("ROC curves comparison")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
