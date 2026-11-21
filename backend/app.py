import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask("SuperKart Stores Sales Forecasting Predictor")

# Load the trained Boston housing model
model = joblib.load("superkart_model.joblib")

# Define a route for the home page
@app.get('/')
def home():
    return "Welcome to the SuperKart Stores Sales Forecasting Predictor API!"

# Define an endpoint to predict sale for a single store
@app.post('/v1/salesforecast')
def predict_store_sales():
    # Get JSON data from the request
    super_kart_data = request.get_json()

        # Extract relevant product and store features from the input data
    sample = {
    'Product_Weight': super_kart_data['Product_Weight'],
    'Product_MRP': super_kart_data['Product_MRP'],
    'Store_Age_Years': super_kart_data['Store_Age_Years'],
    'Product_Id_char': super_kart_data['Product_Id_char'],
    'Product_Type_Category': super_kart_data['Product_Type_Category'],
    'Store_Size': super_kart_data['Store_Size'],
    'Store_Location_City_Type': super_kart_data['Store_Location_City_Type'],
    'Store_Type': super_kart_data['Store_Type']
}

    # Convert the extracted data into a DataFrame
    input_data = pd.DataFrame([sample])

    # Make a prediction using the trained model
    prediction = model.predict(input_data).tolist()[0]

    # Return the prediction as a JSON response
    return jsonify({'Predicted_Product_Store_Sales_Total': prediction})

# Define an endpoint to predict price for a batch of houses
@app.post('/v1/salesforecastbatch')
def predict_house_batch():
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the file into a DataFrame
    input_data = pd.read_csv(file)

    # Make predictions for the batch data
    predictions = model.predict(input_data).tolist()

    # Add predictions to the DataFrame
    input_data['Predicted_Product_Store_Sales_Total'] = predictions

    # Convert results to dictionary
    result = input_data.to_dict(orient="records")

    return jsonify(result)

# Run the Flask app in debug mode
if __name__ == '__main__':
    app.run(debug=True)
