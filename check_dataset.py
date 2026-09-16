import pandas as pd

# Load the dataset
data = pd.read_csv("dataset/car data.csv")

# Display first 5 rows
print("FIRST 5 ROWS:")
print(data.head())

# Display number of rows and columns
print("\nDATASET SHAPE:")
print(data.shape)

# Display column names
print("\nCOLUMN NAMES:")
print(data.columns)

# Display information about the dataset
print("\nDATASET INFORMATION:")
print(data.info())

# Check missing values
print("\nMISSING VALUES:")
print(data.isnull().sum())

# Check duplicate rows
print("\nDUPLICATE ROWS:")
print(data.duplicated().sum())