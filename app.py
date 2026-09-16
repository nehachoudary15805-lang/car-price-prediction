from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved ML model
model = joblib.load("car_price_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get values from the form
    car_name = request.form["Car_Name"]
    year = int(request.form["Year"])
    present_price = float(request.form["Present_Price"])
    driven_kms = int(request.form["Driven_kms"])
    fuel_type = request.form["Fuel_Type"]
    selling_type = request.form["Selling_type"]
    transmission = request.form["Transmission"]
    owner = int(request.form["Owner"])

    # Create input data
    input_data = pd.DataFrame([{
        "Car_Name": car_name,
        "Year": year,
        "Present_Price": present_price,
        "Driven_kms": driven_kms,
        "Fuel_Type": fuel_type,
        "Selling_type": selling_type,
        "Transmission": transmission,
        "Owner": owner
    }])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display prediction
    return render_template(
        "index.html",
        prediction=round(prediction, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)