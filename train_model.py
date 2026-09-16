import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load cleaned dataset
data = pd.read_csv("dataset/cleaned_car_data.csv")


# 2. Select input features
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


# 3. Select target
y = data["Selling_Price"]


# 4. Define categorical and numerical columns
categorical_columns = [
    "Car_Name",
    "Fuel_Type",
    "Selling_type",
    "Transmission"
]

numerical_columns = [
    "Year",
    "Present_Price",
    "Driven_kms",
    "Owner"
]


# 5. Create preprocessing
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


# 6. Create ML model
model = ExtraTreesRegressor(
    n_estimators=200,
    random_state=42
)


# 7. Create complete pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# 8. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# 9. Train the model
pipeline.fit(X_train, y_train)


# 10. Make predictions
y_pred = pipeline.predict(X_test)


# 11. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)


print("MODEL TRAINING COMPLETED!")
print("\nMODEL: Extra Trees Regressor")

print("\nMean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("R2 Score:", r2)