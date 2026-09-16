import pandas as pd
import joblib

# Load the saved model
model = joblib.load("car_price_model.pkl")

# Create sample car details
car = pd.DataFrame([{
    "Car_Name": "swift",
    "Year": 2014,
    "Present_Price": 6.87,
    "Driven_kms": 42450,
    "Fuel_Type": "Diesel",
    "Selling_type": "Dealer",
    "Transmission": "Manual",
    "Owner": 0
}])

# Predict selling price
prediction = model.predict(car)

print("MODEL TEST SUCCESSFUL!")
print("Predicted Selling Price:", prediction[0], "lakhs")