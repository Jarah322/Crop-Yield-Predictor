from flask import Flask, request, render_template, url_for, redirect, jsonify
import numpy as np
import pickle
import os
from datetime import datetime
import json
import random

# Initialize Flask app with enhanced configuration
app = Flask(__name__,
            static_folder='static',
            template_folder='templates',
            static_url_path='/static')

# Load the trained model and preprocessor
try:
    print("\nDebug - Loading model and preprocessor...")
    with open('dtr.pkl', 'rb') as f:
        model = pickle.load(f)
        print(f"Model loaded successfully. Model type: {type(model)}")
    
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
        print(f"Preprocessor loaded successfully. Preprocessor type: {type(preprocessor)}")
except Exception as e:
    print(f"Error loading model or preprocessor: {str(e)}")
    raise e

# Load crop and region data
with open('static/data/crops.json', 'r') as f:
    CROPS = json.load(f)

with open('static/data/regions.json', 'r') as f:
    REGIONS = json.load(f)

# Base yield ranges for different crops (kg/ha)
CROP_BASE_YIELDS = {
    "Maize": (3000, 4000),
    "Rice": (2500, 3500),
    "Wheat": (2000, 3000),
    "Sorghum": (1500, 2500),
    "Soybeans": (2000, 3000),
    "Cassava": (10000, 15000),
    "Sweet potatoes": (8000, 12000),
    "Yams": (7000, 10000),
    "Potatoes": (5000, 8000),
    "Plantains and others": (3000, 5000)
}

# Region-specific multipliers
REGION_MULTIPLIERS = {
    "Northern": 1.0,
    "Upper East": 0.9,
    "Upper West": 0.9,
    "Brong Ahafo": 1.1,
    "Ashanti": 1.2,
    "Eastern": 1.1,
    "Central": 1.1,
    "Western": 1.2,
    "Volta": 1.0,
    "Greater Accra": 0.8
}

def calculate_prediction(region, crop, temperature, rainfall, pesticide):
    # Get base yield range for the crop
    base_min, base_max = CROP_BASE_YIELDS.get(crop, (2000, 3000))
    base_yield = (base_min + base_max) / 2  # Use average of the range
    
    # Apply region multiplier
    region_multiplier = REGION_MULTIPLIERS.get(region, 1.0)
    base_yield *= region_multiplier
    
    # Temperature adjustment (optimal range: 20-30°C)
    if temperature < 20:
        temp_factor = 1 - (20 - temperature) * 0.02
    elif temperature > 30:
        temp_factor = 1 - (temperature - 30) * 0.02
    else:
        temp_factor = 1.0
    
    # Rainfall adjustment (optimal range: 500-2000mm)
    if rainfall < 500:
        rain_factor = 1 - (500 - rainfall) * 0.001
    elif rainfall > 2000:
        rain_factor = 1 - (rainfall - 2000) * 0.0005
    else:
        rain_factor = 1.0
    
    # Pesticide adjustment (optimal range: 1-5 kg/ha)
    if pesticide < 1:
        pest_factor = 1 - (1 - pesticide) * 0.1
    elif pesticide > 5:
        pest_factor = 1 - (pesticide - 5) * 0.05
    else:
        pest_factor = 1.0
    
    # Calculate final prediction
    prediction = base_yield * temp_factor * rain_factor * pest_factor
    
    # Calculate confidence level based on input stability
    confidence = 90  # Base confidence
    
    # Adjust confidence based on input values
    if 20 <= temperature <= 30:
        confidence += 5
    if 500 <= rainfall <= 2000:
        confidence += 5
    if 1 <= pesticide <= 5:
        confidence += 5
    
    # Ensure confidence is within 85-95 range
    confidence = max(85, min(95, confidence))
    
    return prediction, confidence

def predict_yield(region, crop, temperature, rainfall, pesticide):
    """Predict crop yield using the trained model"""
    try:
        # Debug region handling
        print("\nDebug - Region Analysis:")
        print(f"Input Region: {region}")
        print(f"Available Regions: {REGIONS}")
        
        # Verify region exists in our data
        region_exists = False
        for continent, countries in REGIONS.items():
            if region in countries:
                region_exists = True
                print(f"Region found in continent: {continent}")
                break
        
        if not region_exists:
            print("Warning: Region not found in our dataset")
            # Use a default region that exists in our training data
            region = "Ghana"  # or any other country from your training data
            print(f"Using default region: {region}")
        
        # Create input data array with the exact same column order as in the notebook
        current_year = datetime.now().year
        input_data = np.array([[current_year, rainfall, pesticide, temperature, region, crop]], dtype=object)
        
        print(f"\nDebug - Input Data:")
        print(f"Year: {current_year}")
        print(f"Region: {region}")
        print(f"Crop: {crop}")
        print(f"Temperature: {temperature}")
        print(f"Rainfall: {rainfall}")
        print(f"Pesticide: {pesticide}")
        
        # Preprocess the input data
        processed_data = preprocessor.transform(input_data)
        print(f"\nDebug - Processed Data:")
        print(f"Shape: {processed_data.shape}")
        print(f"First 10 features: {processed_data[0][:10]}")
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        print(f"\nDebug - Raw Prediction (hg/ha): {prediction}")
        
        # Convert prediction from hg/ha to kg/ha (1 hg = 0.1 kg)
        prediction_kg = prediction * 0.1
        print(f"Converted Prediction (kg/ha): {prediction_kg}")
        
        # Calculate confidence based on input ranges
        confidence = 90  # Base confidence
        
        # Adjust confidence based on input values
        if 20 <= temperature <= 30:
            confidence += 5
        if 500 <= rainfall <= 2000:
            confidence += 5
        if 1 <= pesticide <= 5:
            confidence += 5
        
        # Ensure confidence is within 85-95 range
        confidence = max(85, min(95, confidence))
        
        print(f"Final Prediction (kg/ha): {prediction_kg}")
        print(f"Confidence: {confidence}")
        
        return prediction_kg, confidence
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None, 0

def generate_insights(region, crop, temperature, rainfall, pesticide, prediction):
    """Generate AI-powered insights based on the prediction"""
    insights = []
    
    # Temperature analysis
    if temperature < 20:
        insights.append(f"The temperature is relatively low for {crop}. Consider methods to maintain optimal growing temperature.")
    elif temperature > 30:
        insights.append(f"The temperature is high. Ensure adequate irrigation and consider shade management.")
    else:
        insights.append("Temperature conditions are favorable for crop growth.")
    
    # Rainfall analysis
    if rainfall < 500:
        insights.append("Rainfall is below optimal levels. Consider supplementary irrigation.")
    elif rainfall > 2000:
        insights.append("High rainfall levels detected. Monitor for potential water-logging and disease risks.")
    else:
        insights.append("Rainfall levels are within acceptable range.")
    
    # Pesticide usage analysis
    if pesticide > 5:
        insights.append("Consider reducing pesticide usage and implementing integrated pest management.")
    elif pesticide < 1:
        insights.append("Monitor crop health closely as pesticide usage is minimal.")
    
    # Region-specific insights
    continent = None
    for cont, countries in REGIONS.items():
        if region in countries:
            continent = cont
            break
    
    if continent:
        if continent == "Africa":
            insights.append("African regions typically have diverse microclimates. Consider local soil conditions and traditional farming practices.")
        elif continent == "Asia":
            insights.append("Asian regions often have intensive farming systems. Consider crop rotation and soil health management.")
        elif continent == "Europe":
            insights.append("European regions typically have well-established agricultural practices. Consider precision farming techniques.")
        elif continent == "North America":
            insights.append("North American regions often have advanced agricultural technology. Consider modern farming equipment and techniques.")
        elif continent == "South America":
            insights.append("South American regions have diverse ecosystems. Consider sustainable farming practices.")
        elif continent == "Oceania":
            insights.append("Oceanic regions often face unique climate challenges. Consider water management and soil conservation.")
    
    return " ".join(insights)

# Root route - AI Assistant Welcome
@app.route('/')
def welcome():
    return render_template('welcome.html')

# Main prediction interface
@app.route('/predictor')
def predictor():
    return render_template('index.html', 
                         crops=CROPS,
                         regions=REGIONS)

# API endpoint for predictions
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("\nDebug - Received API Request:")
        print(f"Request Data: {data}")
        
        # Extract input parameters
        region = data.get('region')
        crop = data.get('crop')
        temperature = float(data.get('temperature'))
        rainfall = float(data.get('rainfall'))
        pesticide = float(data.get('pesticide'))
        
        # Input validation
        if not all([region, crop, temperature, rainfall, pesticide]):
            return jsonify({'error': 'All fields are required'}), 400
        
        # Make prediction using the trained model
        prediction, confidence = predict_yield(region, crop, temperature, rainfall, pesticide)
        
        if prediction is None:
            return jsonify({'error': 'Failed to make prediction'}), 500
        
        # Generate insights
        insights = generate_insights(region, crop, temperature, rainfall, pesticide, prediction)
        
        response = {
            'prediction': round(prediction, 2),  # Round to 2 decimal places
            'confidence': confidence,
            'insights': insights
        }
        print(f"\nDebug - API Response: {response}")
        
        return jsonify(response)
    
    except Exception as e:
        print(f"API Error: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)