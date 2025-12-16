# Energy Consumption Modelling using EIA Power System Data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score


# -------------------------------
# Data Loading
# -------------------------------

df = pd.read_csv(
    "electricity-data.csv",
    engine="python",
    on_bad_lines="skip"
)

# TotalConsumptionBtu units: Million MMBtu


# -------------------------------
# Initial Exploration (kept minimal for script form)
# -------------------------------

# NOTE: The following operations were used during exploration in the notebook
# and are intentionally omitted from printing in this script version:
# df.describe()
# value_counts() on categorical columns


# -------------------------------
# Feature Selection & Encoding
# -------------------------------

# SINCE LOCATION AND STATE IS BASICALLY THE SAME FEATURE BUT LOCATION IS ABBREVIATED,
# IT IS BETTER TO DROP ONE OF THE FEATURE ENTIRELY

df = df.drop("location", axis=1)

# Get dummy variables to convert non numeric features into numeric (binary) ones

df = pd.get_dummies(df, columns=["state"], drop_first=True, dtype=int)
df = pd.get_dummies(df, columns=["sector"], drop_first=True, dtype=int)
df = pd.get_dummies(df, columns=["fuel"], drop_first=True, dtype=int)


# -------------------------------
# Correlation Analysis
# -------------------------------

plt.figure(figsize=(5, 15))

corr_series = (
    df.corr(numeric_only=True)["totalConsumptionBtu"]
    .drop("totalConsumptionBtu")
    .sort_values(ascending=False)
)

sns.heatmap(corr_series.to_frame(), cmap="viridis", annot=True)
plt.title("Feature Correlation with Total Fuel Consumption")

# Save plot instead of blocking execution
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.close()

# Total Consumption is highly correlated with generation feature,
# then it faces a steep decline across all other features.
# This will be taken into consideration.


# -------------------------------
# Target and Feature Split
# -------------------------------

y = df["totalConsumptionBtu"]
X = df.drop(["totalConsumptionBtu", "period"], axis=1)

# Structural features only (generation removed)
X_no_gen = X.drop(columns=["generation"])


# -------------------------------
# Ridge Regression
# -------------------------------

# With Generation Feature

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

r2_ridge_with_generation = r2_score(y_test, model.predict(X_test))
print("R² (Ridge | with generation):", r2_ridge_with_generation)


# Without Generation Feature

X_train, X_test, y_train, y_test = train_test_split(
    X_no_gen, y, test_size=0.2, random_state=42
)

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

r2_ridge_without_generation = r2_score(y_test, model.predict(X_test))
print("R² (Ridge | without generation):", r2_ridge_without_generation)


# Since fuel consumption (Btu) is directly related as seen in correlation heatmap
# and through formula:
# Fuel Consumption (MMBtu) = Generation (MWh) × (Heat Rate (Btu/kWh) / 1000)
# r^2 score shoots up significantly when generation is included.


# -------------------------------
# Random Forest Regression
# -------------------------------

# RF Regressor with Generation

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=3,
    max_features=1.0,
    bootstrap=True,
    n_jobs=1,
    random_state=42
)

rf.fit(X_train, y_train)

r2_rf_with_generation = r2_score(y_test, rf.predict(X_test))
print("R² (RF | with generation):", r2_rf_with_generation)


# RF Regressor without Generation

X_train, X_test, y_train, y_test = train_test_split(
    X_no_gen, y, test_size=0.2, random_state=42
)

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=3,
    max_features=1.0,
    bootstrap=True,
    n_jobs=1,
    random_state=42
)

rf.fit(X_train, y_train)

r2_rf_without_generation = r2_score(y_test, rf.predict(X_test))
print("R² (RF | without generation):", r2_rf_without_generation)


# -------------------------------
# Grid Search RF Regressor (Validation)
# -------------------------------

# With Generation Feature

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_leaf": [1, 5, 10],
    "max_features": ["sqrt", "log2"],
}

base_rf = RandomForestRegressor(n_estimators=100, random_state=42)

grid_rf = GridSearchCV(
    base_rf,
    param_grid,
    cv=2,
    scoring="r2",
    verbose=1
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

grid_rf.fit(X_train, y_train)

best_rf = grid_rf.best_estimator_
best_rf.fit(X_train, y_train)

print("R² (RF GridSearch | with generation):", r2_score(y_test, best_rf.predict(X_test)))


# Without Generation Feature

X_train, X_test, y_train, y_test = train_test_split(
    X_no_gen, y, test_size=0.2, random_state=42
)

grid_rf = GridSearchCV(
    base_rf,
    param_grid,
    cv=2,
    scoring="r2",
    verbose=1
)

grid_rf.fit(X_train, y_train)

best_rf = grid_rf.best_estimator_
best_rf.fit(X_train, y_train)

print("R² (RF GridSearch | without generation):", r2_score(y_test, best_rf.predict(X_test)))


# -------------------------------
# Results Summary
# -------------------------------

results = pd.DataFrame({
    "model": ["Ridge", "Ridge", "Random Forest", "Random Forest"],
    "generation_included": [True, False, True, False],
    "r2_score": [
        r2_ridge_with_generation,
        r2_ridge_without_generation,
        r2_rf_with_generation,
        r2_rf_without_generation,
    ],
})

print("\nFinal Model Comparison:\n")
print(results)
