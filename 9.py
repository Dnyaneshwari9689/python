# PR11: Advanced Machine Learning and Model Optimization
# SVM, Random Forest, AdaBoost, Cross Validation and Grid Search

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score

# 1. Load CSV dataset
data = pd.read_csv("breast_cancer.csv")
X = data.drop("target", axis=1)
y = data["target"]

print("Dataset Shape:", data.shape)
print("First 5 Records:")
print(data.head())

# 2. Split dataset: 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Define models
models = {
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC())
    ]),
    "Random Forest": RandomForestClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42)
}

# 4. Evaluate models before optimization
before_results = []
print("\\n========== BEFORE OPTIMIZATION ==========")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

    before_results.append({
        "Model": name,
        "Test Accuracy Before (%)": test_acc * 100,
        "CV Accuracy Before (%)": cv_scores.mean() * 100
    })

    print(f"\\nModel: {name}")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print(f"5-Fold CV Accuracy: {cv_scores.mean() * 100:.2f}%")

# 5. Define hyperparameter grids
param_grids = {
    "SVM": {
        "model__C": [0.1, 1, 10],
        "model__kernel": ["linear", "rbf"],
        "model__gamma": ["scale", "auto"]
    },
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10]
    },
    "AdaBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 1.0]
    }
}

# 6. Grid Search with 5-fold cross validation
after_results = []
best_models = {}

print("\\n========== AFTER OPTIMIZATION ==========")

for name, model in models.items():
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grids[name],
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_models[name] = best_model
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    after_results.append({
        "Model": name,
        "Best CV Accuracy (%)": grid.best_score_ * 100,
        "Test Accuracy After (%)": test_acc * 100
    })

    print(f"\\nModel: {name}")
    print("Best Parameters:", grid.best_params_)
    print(f"Best 5-Fold CV Accuracy: {grid.best_score_ * 100:.2f}%")
    print(f"Test Accuracy After Tuning: {test_acc * 100:.2f}%")

# 7. Compare before and after
before = pd.DataFrame(before_results)
after = pd.DataFrame(after_results)
comparison = before.merge(after, on="Model")

print("\\n========== FINAL COMPARISON ==========")
print(comparison.round(2).to_string(index=False))
