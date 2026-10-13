
import pandas as pd

# Load the dataset
data = pd.read_csv("dataset/house_prices.csv")

# Display the first five rows
print("\n" + "=" * 60)
print("HOUSE PRICE DATASET - FIRST FIVE ROWS")
print("=" * 60)
print(data.head().to_string(index=False))

# Display dataset dimensions
print("\n" + "=" * 60)
print("DATASET SHAPE")
print("=" * 60)
print(f"Total records: {data.shape[0]}")
print(f"Total columns: {data.shape[1]}")

# Display column names and data types
print("\n" + "=" * 60)
print("COLUMN NAMES AND DATA TYPES")
print("=" * 60)
print(data.dtypes)

# Check for missing values
print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
print(data.isnull().sum())

# Display statistical summary
print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)
print(data.describe().round(2).to_string())

print("\nCATEGORICAL COLUMN SUMMARY")
print("=" * 60)

categorical_columns = data.select_dtypes(include=["object", "str"]).columns

for column in categorical_columns:
    print(f"\n{column}:")
    print(data[column].value_counts())

print("\nDataset inspection completed successfully!")




# ============================================================
# HOUSE PRICE PREDICTION - LINEAR REGRESSION
# ============================================================

import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("\n" + "=" * 60)
print("HOUSE PRICE PREDICTION MODEL")
print("=" * 60)

# Separate input features and target
X = data.drop(columns=["price"])
y = data["price"]

# Identify numerical and categorical columns
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

print("\nNumerical Features:", numeric_features)
print("Categorical Features:", categorical_features)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"\nTraining records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")

# Encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("numerical", "passthrough", numeric_features)
    ]
)

# Create the machine-learning pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

# Train the model
model.fit(X_train, y_train)

print("\nModel training completed successfully!")

# Predict house prices for the test data
y_pred = model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"Mean Absolute Error (MAE): ₹{mae:,.2f}")
print(f"Root Mean Squared Error (RMSE): ₹{rmse:,.2f}")
print(f"R-squared (R²): {r2:.4f}")

# Compare actual and predicted prices
comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred.round(0).astype(int)
})

print("\n" + "=" * 60)
print("ACTUAL VS PREDICTED PRICES - FIRST 10 HOUSES")
print("=" * 60)

print(comparison.head(10).to_string(index=False))

# Create a directory for saved model files
os.makedirs("model", exist_ok=True)

# Save the trained model
import joblib
joblib.dump(model, "model/house_price_model.pkl")

print("\nTrained model saved to model/house_price_model.pkl")





# ============================================================
# ACTUAL VS PREDICTED PRICE VISUALIZATION
# ============================================================

import matplotlib.pyplot as plt
import os

os.makedirs("charts", exist_ok=True)

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.7)

# Reference line for perfect predictions
minimum_price = min(y_test.min(), y_pred.min())
maximum_price = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum_price, maximum_price],
    [minimum_price, maximum_price],
    linestyle="--",
    color="red",
    label="Perfect Prediction"
)

plt.title("Actual vs Predicted House Prices")
plt.xlabel("Actual Price (₹)")
plt.ylabel("Predicted Price (₹)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("charts/actual_vs_predicted.png", dpi=300)
plt.show()

print("\nActual vs predicted chart saved successfully!")

