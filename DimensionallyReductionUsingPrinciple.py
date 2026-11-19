# PR10: Dimensionality Reduction using PCA
# Breast Cancer Dataset

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, confusion_matrix

# ---------------------------------------
# 1. Load Dataset
# ---------------------------------------

data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset Shape:", X.shape)
print("First 5 Records:")
print(pd.DataFrame(X, columns=data.feature_names).head())

# ---------------------------------------
# 2. Split Dataset
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------
# 3. Feature Scaling
# ---------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeatures Before PCA:", X_train.shape[1])

# ---------------------------------------
# 4. Classification Before PCA
# ---------------------------------------

model1 = LogisticRegression(max_iter=5000)

model1.fit(X_train_scaled, y_train)

y_pred1 = model1.predict(X_test_scaled)

accuracy1 = accuracy_score(y_test, y_pred1)

print("\n--- Before PCA ---")
print("Accuracy:", accuracy1)
print("Precision:", precision_score(y_test, y_pred1))
print("Recall:", recall_score(y_test, y_pred1))
print("F1 Score:", f1_score(y_test, y_pred1))

# ---------------------------------------
# 5. Apply PCA
# ---------------------------------------

pca = PCA(n_components=0.95)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("\nFeatures After PCA:", X_train_pca.shape[1])

print("Explained Variance:",
      sum(pca.explained_variance_ratio_))

# ---------------------------------------
# 6. Classification After PCA
# ---------------------------------------

model2 = LogisticRegression(max_iter=5000)

model2.fit(X_train_pca, y_train)

y_pred2 = model2.predict(X_test_pca)

accuracy2 = accuracy_score(y_test, y_pred2)

print("\n--- After PCA ---")
print("Accuracy:", accuracy2)
print("Precision:", precision_score(y_test, y_pred2))
print("Recall:", recall_score(y_test, y_pred2))
print("F1 Score:", f1_score(y_test, y_pred2))

# ---------------------------------------
# 7. Confusion Matrix
# ---------------------------------------

print("\nConfusion Matrix Before PCA:")
print(confusion_matrix(y_test, y_pred1))

print("\nConfusion Matrix After PCA:")
print(confusion_matrix(y_test, y_pred2))

# ---------------------------------------
# 8. PCA Visualization (2D)
# ---------------------------------------

pca_2d = PCA(n_components=2)

X_2d = pca_2d.fit_transform(
    scaler.fit_transform(X)
)

plt.figure(figsize=(8, 5))

plt.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    c=y,
    cmap='viridis'
)

plt.title("PCA - 2D Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Cancer Class")
plt.show()

# ---------------------------------------
# 9. Compare Classification Accuracy
# ---------------------------------------

models = ['Before PCA', 'After PCA']
accuracies = [accuracy1 * 100, accuracy2 * 100]

plt.figure(figsize=(7, 5))

plt.bar(models, accuracies)

plt.title("Classification Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 105)
plt.show()

# ---------------------------------------
# 10. Final Comparison
# ---------------------------------------

print("\n--- Final Comparison ---")

print("Original Features:", X.shape[1])
print("Reduced Features:", X_train_pca.shape[1])

print("Accuracy Before PCA:", accuracy1 * 100, "%")
print("Accuracy After PCA:", accuracy2 * 100, "%")