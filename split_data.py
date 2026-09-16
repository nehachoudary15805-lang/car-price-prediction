import pandas as pd
from sklearn.model_selection import train_test_split

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

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Total data:", len(X))
print("Training data:", len(X_train))
print("Testing data:", len(X_test))

print("\nTraining target values:", len(y_train))
print("Testing target values:", len(y_test))

print("\nData splitting completed successfully!")