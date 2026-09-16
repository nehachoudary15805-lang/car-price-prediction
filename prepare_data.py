import pandas as pd

# Load cleaned dataset
data = pd.read_csv("dataset/cleaned_car_data.csv")

# Select input features
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

# Select target
y = data["Selling_Price"]

# Display input features
print("INPUT FEATURES:")
print(X.head())

# Display target values
print("\nTARGET VALUES:")
print(y.head())

# Display shapes
print("\nINPUT FEATURES SHAPE:")
print(X.shape)

print("\nTARGET SHAPE:")
print(y.shape)

# Display data types
print("\nINPUT FEATURE DATA TYPES:")
print(X.dtypes)

# Display categorical columns
print("\nCATEGORICAL COLUMNS:")
print(X[[
    "Car_Name",
    "Fuel_Type",
    "Selling_type",
    "Transmission"
]].head())