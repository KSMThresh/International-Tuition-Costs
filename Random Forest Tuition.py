import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Step 1: Load dataset
file_path = r"C:\Users\willc\OneDrive\Documents\Kaggle Datasets\International_Education_Costs.csv"
df = pd.read_csv(file_path).dropna()

# Step 2: Define features and target
X = df.drop(columns=["Tuition_USD", "University"])  # Dropping target and unique identifier
y = df["Tuition_USD"]

# Step 3: Set categorical and numerical features
categorical_features = ["Country", "City", "Level", "Program"]
numerical_features = ["Duration_Years", "Living_Cost_Index", "Rent_USD", "Visa_Fee_USD", "Insurance_USD", "Exchange_Rate"]

# Step 4: Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough"  # Pass numerical features through unchanged
)

# Step 5: Create full pipeline with RandomForestRegressor
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train model
pipeline.fit(X_train, y_train)

# Step 8: Predictions and performance
y_pred = pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R²: {r2:.4f}")

# Actual vs. predicted
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel("Actual Tuition (USD)")
plt.ylabel("Predicted Tuition (USD)")
plt.title("Random Forest: Actual vs Predicted Tuition")
plt.grid(True)
plt.tight_layout()
plt.show()

# Residuals Graph
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Tuition")
plt.ylabel("Residuals")
plt.title("Residuals vs. Predicted Tuition")
plt.grid(True)
plt.tight_layout()
plt.show()

# Results: The Random Forest Regression model has yielded better results than the Linear Regression from earlier
#  The actual vs predicted plot has shown to have it's predicted values to align more closely than the linear regression
#  It is also worth pointing out that the Random Forest has a higher coefficient of determination than the linear regression
#  Residuals also look evenly distributed but there is no clear difference between which one is more "random" compared to the linear regression
#  In conclusion, Random Forest is better for this dataset rather than linear regression