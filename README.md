# ZongoVation Crop Yield Prediction

## Overview
This project is a Flask-based web application that predicts crop yields based on various environmental and agricultural inputs. It uses a trained model and preprocessor to provide predictions and insights for different crops and regions.

## Features
- Crop yield prediction based on region, crop type, temperature, rainfall, and pesticide usage
- AI-powered insights for agricultural practices
- User-friendly web interface
- RESTful API for predictions

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Final-Project-ZongoVation
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the model and preprocessor files (`dtr.pkl` and `preprocessor.pkl`) are in the root directory.

## Usage
1. Start the Flask application:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to `http://localhost:5000` to access the web interface.

3. Use the predictor interface to input your data and receive predictions.

## API Endpoints
- **GET /**: Welcome page
- **GET /predictor**: Main prediction interface
- **POST /api/predict**: API endpoint for predictions

## Project Structure
- `app.py`: Main application file
- `static/`: Static files (CSS, JavaScript, data)
- `templates/`: HTML templates
- `dtr.pkl`: Trained model
- `preprocessor.pkl`: Preprocessor for data transformation

## Contributing
Feel free to submit issues and enhancement requests!

## License
This project is licensed under the MIT License - see the LICENSE file for details. 