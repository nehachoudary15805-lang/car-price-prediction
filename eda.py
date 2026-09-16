import pandas as pd

# Load cleaned dataset
data = pd.read_csv("dataset/cleaned_car_data.csv")

# Display first 5 rows
print("FIRST 5 ROWS:")
print(data.head())

# Display dataset shape
print("\nDATASET SHAPE:")
print(data.shape)

# Display basic statistics
print("\nBASIC STATISTICS:")
print(data.describe())

# Display unique values
print("\nFUEL TYPES:")
print(data["Fuel_Type"].value_counts())

print("\nSELLING TYPES:")
print(data["Selling_type"].value_counts())

print("\nTRANSMISSION TYPES:")
print(data["Transmission"].value_counts())

print("\nOWNER TYPES:")
print(data["Owner"].value_counts())

# Number of unique car names
print("\nNUMBER OF UNIQUE CAR NAMES:")
print(data["Car_Name"].nunique())