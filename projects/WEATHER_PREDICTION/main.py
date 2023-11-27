import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import joblib

def train_weather_model(df, features, target):

    # Drop rows with missing values in the selected features and target
    df = df.dropna(subset=features + [target])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=0.2, random_state=42
    )

    # Initialize and train a machine learning model (Random Forest Classifier as an example)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Evaluate the model (optional)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy}")

    # Save the trained model using joblib
    joblib.dump(model, 'weather_model.joblib')

    return model

def predict_weather_conditions(model, input_data):

    # Make predictions on the input data
    predictions = model.predict(input_data)

    return predictions

def get_user_input_and_predict(model):

    # Get input parameters from the user
    temp_c = float(input("Enter Temperature in Celsius: "))
    dew_point_temp_c = float(input("Enter Dew Point Temperature in Celsius: "))
    rel_humidity = int(input("Enter Relative Humidity (%): "))
    wind_speed_kmh = int(input("Enter Wind Speed in km/h: "))
    visibility_km = float(input("Enter Visibility in km: "))
    press_kpa = float(input("Enter Atmospheric Pressure in kPa: "))

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
    predicted_weather = predict_weather_conditions(model, input_data)

    return predicted_weather

# Define features and target_column
features = ['Temp_C', 'Dew Point Temp_C', 'Rel Hum_%', 'Wind Speed_km/h', 'Visibility_km', 'Press_kPa']
target_column = 'Weather'

# Load your dataset (assuming df is already loaded)

# Train the model
trained_model = train_weather_model(df, features, target_column)

# Create a pipeline with the model
pipeline = Pipeline([
    ('model', trained_model)
])

# Save the pipeline using joblib
joblib.dump(pipeline, 'weather_model_pipeline.joblib')

# Load the pipeline
loaded_pipeline = joblib.load('weather_model_pipeline.joblib')

# Get user input and make predictions using the loaded pipeline
predicted_weather_loaded_pipeline = get_user_input_and_predict(loaded_pipeline['model'])

print("Predicted Weather Condition (Loaded Pipeline):", predicted_weather_loaded_pipeline[0])

