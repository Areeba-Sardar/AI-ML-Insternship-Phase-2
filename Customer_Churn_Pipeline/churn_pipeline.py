import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, classification_report


# ==========================
# 1. Load Dataset
# ==========================

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset Loaded Successfully")
print(df.head())


# ==========================
# 2. Data Cleaning
# ==========================

# Remove customer ID
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

# Convert target column
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})


# ==========================
# 3. Features and Target
# ==========================

X = df.drop("Churn", axis=1)
y = df["Churn"]


# ==========================
# 4. Find Column Types
# ==========================

numerical_cols = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_cols = X.select_dtypes(
    include=["object"]
).columns


# ==========================
# 5. Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================
# 6. Preprocessing Pipeline
# ==========================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_cols
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_cols
        )
    ]
)


# ==========================
# 7. Logistic Regression
# ==========================

log_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

log_pipeline.fit(X_train, y_train)

log_pred = log_pipeline.predict(X_test)

print("\n========== Logistic Regression ==========")
print("Accuracy:",
      accuracy_score(y_test, log_pred))

print(classification_report(
    y_test,
    log_pred
))


# ==========================
# 8. Random Forest Pipeline
# ==========================

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        random_state=42
    ))
])


# ==========================
# 9. GridSearchCV
# ==========================

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [5, 10, None]
}

grid_search = GridSearchCV(
    rf_pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)


# ==========================
# 10. Best Model
# ==========================

best_model = grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)


# ==========================
# 11. Evaluation
# ==========================

rf_pred = best_model.predict(X_test)

print("\n========== Random Forest ==========")
print("Accuracy:",
      accuracy_score(y_test, rf_pred))

print(classification_report(
    y_test,
    rf_pred
))


# ==========================
# 12. Save Model
# ==========================

joblib.dump(
    best_model,
    "churn_pipeline.pkl"
)

print("\nModel Saved Successfully")
print("File Name: churn_pipeline.pkl")