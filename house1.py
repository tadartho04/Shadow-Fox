import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("HousingData.csv")

# Show first 5 rows
print("First 5 rows:")
print(df.head())

# Dataset shape
print("\nDataset shape (rows, columns):")
print(df.shape)

# Column names
print("\nColumn names:")
print(df.columns)

# Dataset info
print("\nDataset information:")
print(df.info())

# Check missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# =========================
# HOUR 2: HANDLE MISSING VALUES
# =========================

# Check missing values again
print("Missing values before cleaning:")
print(df.isnull().sum())

# Fill missing values with mean
df["AGE"].fillna(df["AGE"].mean(), inplace=True)
df["LSTAT"].fillna(df["LSTAT"].mean(), inplace=True)

# Verify cleaning
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# =========================
# HOUR 3: TRAIN-TEST SPLIT
# =========================

# Separate features and target
X = df.drop("MEDV", axis=1)
y = df["MEDV"]

print("Shape of X (features):", X.shape)
print("Shape of y (target):", y.shape)

# Split into training and testing data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nAfter train-test split:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# =========================
# HOUR 4: FEATURE SCALING
# =========================

from sklearn.preprocessing import StandardScaler

# Create scaler object
scaler = StandardScaler()

# Fit on training data and transform
X_train = scaler.fit_transform(X_train)

# Transform test data
X_test = scaler.transform(X_test)

print("Feature scaling completed.")
print("X_train mean (approx):", X_train.mean())
print("X_train std (approx):", X_train.std())

# =========================
# HOUR 5: MODEL COMPARISON & VISUALIZATION
# =========================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -------------------------
# Train Random Forest Model
# -------------------------
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

# Predictions
y_pred_rf = rf.predict(X_test)

# -------------------------
# Evaluation Metrics
# -------------------------
r2 = r2_score(y_test, y_pred_rf)
mae = mean_absolute_error(y_test, y_pred_rf)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print("===== Random Forest Model Performance =====")
print("R2 Score:", round(r2, 3))
print("MAE:", round(mae, 3))
print("RMSE:", round(rmse, 3))

# -------------------------
# Visualization
# -------------------------
plt.figure()
plt.scatter(y_test, y_pred_rf)
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices (Random Forest)")
plt.show()