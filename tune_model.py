import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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


# Extra Trees model
model = ExtraTreesRegressor(
    random_state=42
)


# Complete pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Parameters to test
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}


# Grid Search
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)


print("TUNING MODEL...")
print("Please wait...")


# Train and find best parameters
grid_search.fit(X_train, y_train)


# Best model
best_model = grid_search.best_estimator_


# Prediction
y_pred = best_model.predict(X_test)


# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)


print("\nMODEL TUNING COMPLETED!")

print("\nBEST PARAMETERS:")
print(grid_search.best_params_)

print("\nBEST CROSS-VALIDATION RMSE:")
print(-grid_search.best_score_)

print("\nTEST SET RESULTS:")
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("R2 Score:", r2)