import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def load_model():
    # Load the pre-trained model
    model = joblib.load('weather_model.joblib')
    return model

def predict_weather_conditions(model, input_data):
    # Make predictions on the input data
    predictions = model.predict(input_data)
    return predictions[0]

def main():
    # Load the pre-trained model
    model = load_model()

    # Add a title to your app
    st.title("Weather Prediction App")

    # Get user input
    temp_c = st.slider("Temperature in Celsius", min_value=-10.0, max_value=40.0, value=20.0)
    dew_point_temp_c = st.slider("Dew Point Temperature in Celsius", min_value=-10.0, max_value=30.0, value=15.0)
    rel_humidity = st.slider("Relative Humidity (%)", min_value=0, max_value=100, value=50)
    wind_speed_kmh = st.slider("Wind Speed in km/h", min_value=0, max_value=50, value=10)
    visibility_km = st.slider("Visibility in km", min_value=0.1, max_value=50.0, value=10.0)
    press_kpa = st.slider("Atmospheric Pressure in kPa", min_value=90.0, max_value=110.0, value=101.0)

    # Create a DataFrame with user input
    input_data = pd.DataFrame({
        'Temp_C': [temp_c],
        'Dew Point Temp_C': [dew_point_temp_c],
        'Rel Hum_%': [rel_humidity],
        'Wind Speed_km/h': [wind_speed_kmh],
        'Visibility_km': [visibility_km],
        'Press_kPa': [press_kpa],
    })

    # Make predictions
    if st.button("Predict Weather"):
        predicted_weather = predict_weather_conditions(model, input_data)
        st.success(f"Predicted Weather Condition: {predicted_weather}")

if __name__ == '__main__':
    main()

