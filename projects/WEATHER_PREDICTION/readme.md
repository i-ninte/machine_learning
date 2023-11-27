# Weather Prediction App

## Overview

This is a simple Streamlit web application for predicting weather conditions based on user input. The application uses a pre-trained machine learning model to make predictions.

## Features

- Predicts weather conditions based on user input.
- User-friendly interface with sliders for input parameters.
- Utilizes a Random Forest Classifier for prediction.

## Prerequisites

Make sure you have Python installed in your environment. You can install the required packages using the following command:

```bash
pip install streamlit pandas scikit-learn joblib
```
# run the app
```
streamlit run app.py
```
# Usage
Adjust the sliders to input weather conditions.
Click the "Predict Weather" button to see the model's prediction.

# Model Information
The app uses a Random Forest Classifier for predicting weather conditions. The model was trained using data with features such as temperature, humidity, wind speed, visibility, and atmospheric pressure.


