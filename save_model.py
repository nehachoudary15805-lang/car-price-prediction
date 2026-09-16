import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor


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


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numerical",
            "passthrough",
            numerical_columns
        )
    ]
)


# Original Extra Trees model
model = ExtraTreesRegressor(
    n_estimators=200,
    random_state=42
)


# Create complete pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# Train the model using the complete dataset
pipeline.fit(X, y)


# Save the trained model
joblib.dump(pipeline, "car_price_model.pkl")


print("MODEL TRAINED SUCCESSFULLY!")
print("MODEL SAVED SUCCESSFULLY!")
print("File created: car_price_model.pkl")