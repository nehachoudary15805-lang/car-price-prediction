import pandas as pd

# Load dataset
data = pd.read_csv("dataset/car data.csv")

print("Original dataset shape:", data.shape)

# Check duplicate rows
print("\nNumber of duplicate rows:", data.duplicated().sum())

# Remove duplicate rows
data = data.drop_duplicates()

print("Dataset shape after removing duplicates:", data.shape)

# Check missing values
print("\nMissing values:")
print(data.isnull().sum())

# Display cleaned dataset
print("\nCleaned dataset:")
print(data.head())

# Display information
print("\nDataset information:")
data.info()

data.to_csv("dataset/cleaned_car_data.csv", index=False)

print("\nCleaned dataset saved successfully!")