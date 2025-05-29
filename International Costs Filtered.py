import pandas as pd
import statsmodels.api as sm

# Step 1: Load the dataset
file_path = r"C:\Users\willc\OneDrive\Documents\Kaggle Datasets\International_Education_Costs.csv"
df = pd.read_csv(file_path).dropna()

# Step 2: Define features and target
X = df[["Country", "City", "Level", "Program", "Duration_Years",
        "Living_Cost_Index", "Rent_USD", "Visa_Fee_USD", "Insurance_USD", "Exchange_Rate"]]
y = df["Tuition_USD"]

# Step 3: One-hot encode categorical features
X_encoded = pd.get_dummies(X, drop_first=True)

# Step 4: Add intercept
X_encoded = sm.add_constant(X_encoded).astype(float)

# Step 5: Fit full model
model_full = sm.OLS(y, X_encoded).fit()

# Step 6: Filter out predictors with p > 0.05
summary_df = model_full.summary2().tables[1]
insignificant_predictors = summary_df[summary_df['P>|t|'] > 0.05]
X_reduced = X_encoded.drop(columns=insignificant_predictors.index)

# Step 7: Fit reduced model
model_reduced = sm.OLS(y, X_reduced).fit()

# Step 8: Output summary and R²
print("== Reduced Model Summary ==")
print(model_reduced.summary())

# Print R² and Adjusted R² separately
print(f"\nR²: {model_reduced.rsquared:.4f}")
print(f"Adjusted R²: {model_reduced.rsquared_adj:.4f}")
