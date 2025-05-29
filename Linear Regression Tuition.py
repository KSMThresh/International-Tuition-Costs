import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Load the Dataset
file_path = r"C:\Users\willc\OneDrive\Documents\Kaggle Datasets\International_Education_Costs.csv"
df = pd.read_csv(file_path)
print(df.head())

# Cleaning the data
df_clean = df.dropna()

# Define the features and target
X = df_clean.drop(columns=["Tuition_USD", "University", "Program"]) 
y = df_clean["Tuition_USD"]

# Setting the categorical and numerical features
categorical_features = ["Country", "City", "Level"]
numerical_features = ["Duration_Years", "Living_Cost_Index", "Rent_USD", "Visa_Fee_USD", "Insurance_USD", "Exchange_Rate"]

# Setting the pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ],
    remainder="passthrough"  # Leave numerical columns as-is
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Making Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
pipeline.fit(X_train, y_train)

# Assessing the predictions and performance
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(mse)

#Printing the regression equations
regressor = pipeline.named_steps["regressor"]

onehot_feature_names = pipeline.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out(categorical_features)
all_feature_names = list(onehot_feature_names) + numerical_features

coefficients = pd.Series(regressor.coef_, index=all_feature_names)

intercept = regressor.intercept_

print(coefficients.head(), intercept)

# Coefficient of Determination
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")

# Predicted vs Actual
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel("Actual Tuition (USD)")
plt.ylabel("Predicted Tuition (USD)")
plt.title("Actual vs. Predicted Tuition Values")
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

# Example
new_input = pd.DataFrame([{
    "Country": "Canada",
    "City": "Toronto",
    "Level": "Master",
    "Program": "Business Analytics",
    "Duration_Years": 2,
    "Living_Cost_Index": 72.5,
    "Rent_USD": 1600,
    "Visa_Fee_USD": 235,
    "Insurance_USD": 900,
    "Exchange_Rate": 1.35
}])

# Use the trained pipeline to predict tuition
predicted_tuition = pipeline.predict(new_input)
print(f"Predicted Tuition (USD): ${predicted_tuition[0]:,.2f}")

# Note: I have filterred out any of the predictors that weren't "statistically sigificant."
#   After running another linear regression model on those predictors, the R^2 value was 0.6156
#   As a result, I've concluded that this model would be more ideal to work off of
#   You can see that model in the same respository under "International Costs Filtered.py"