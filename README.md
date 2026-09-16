# Car Price Prediction Using Machine Learning

## About the Project

This project predicts the selling price of a car using Machine Learning. The prediction is based on details such as car name, manufacturing year, present price, driven kilometers, fuel type, selling type, transmission, and previous owners.

The project includes a web application where users can enter car details and get the predicted selling price.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Flask
- HTML
- CSS
- Matplotlib
- Seaborn
- Joblib

## Machine Learning Model

The project uses the Extra Trees Regressor algorithm for predicting the car selling price.

Categorical data is converted using One-Hot Encoding, and preprocessing and prediction are handled through a Machine Learning pipeline.

## Input Features

- Car Name
- Manufacturing Year
- Present Price
- Driven Kilometers
- Fuel Type
- Selling Type
- Transmission
- Previous Owners

## Project Workflow

Car Details → HTML/CSS Frontend → Flask Backend → Data Preprocessing → Extra Trees Regressor → Predicted Selling Price

## Dataset

The project uses a car price dataset containing information about different cars and their selling prices.

The target variable is Selling_Price.

## How to Run the Project

1. Open the project folder in VS Code.
2. Activate the virtual environment.
3. Install the required libraries.
4. Run the Flask application using:

python app.py

5. Open the following address in your browser:

http://127.0.0.1:5000

## Result

The web application takes the car details entered by the user and displays the predicted selling price in lakhs.

## Project Structure

CAR_PRICE_PREDICTION/
│
├── dataset/
├── static/
├── templates/
├── app.py
├── car_price_model.pkl
├── train_model.py
├── compare_models.py
├── tune_model.py
└── other Python files
