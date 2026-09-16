import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load cleaned dataset
data = pd.read_csv("dataset/cleaned_car_data.csv")

# Input features
X = data[[
    "Car_Name",
    "Year",
    "Present_Price",
    "Driven_kms",
    "Fuel_Type",
    "Selling_type",
    "Transmission",
    "Owner"
]]

# Target
y = data["Selling_Price"]

# Categorical columns
categorical_columns = [
    "Car_Name",
    "Fuel_Type",
    "Selling_type",
    "Transmission"
]

# Numerical columns
numerical_columns = [
    "Year",
    "Present_Price",
    "Driven_kms",
    "Owner"
]

# Create encoder
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         categorical_columns),
        ("numerical", "passthrough", numerical_columns)
    ]
)

# Encode the categorical columns
X_encoded = preprocessor.fit_transform(X)

print("Original input shape:", X.shape)
print("Encoded input shape:", X_encoded.shape)

print("\nEncoding completed successfully!")