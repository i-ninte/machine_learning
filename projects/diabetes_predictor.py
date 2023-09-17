import pandas as pd
import numpy as np
#reading the dataset
data = pd.read_csv("diabetes.csv")

# viewing first 5 rows
data.head()

#checking the number of records of the dataset
data.shape

# checking the data types 
data.dtypes

#describing the data
data.describe()

# CLEANING THE DATA
# checking for duplicates 
data_duplicate= data[data.duplicated()]
# finding the sum of the duplicates
sum(data.duplicated())




# Visualization 
import matplotlib.pyplot as plt
plt.scatter(data['SkinThickness'], data['Insulin'])
plt.xscale('log')
plt.xlabel('SkinThickness')
plt.ylabel('insulin')
plt.title('relation between skin thickness and insulin')
plt.show()





# building the model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Data Preparation
data = pd.read_csv("diabetes.csv")
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Step 4: Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Selection
model = LogisticRegression()  # You can choose another classifier here

# Step 6: Model Training
model.fit(X_train, y_train)

# Step 7: Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(report)


# Import necessary libraries
import pandas as pd

# Creating new dataframe for patient

new_patient_data = pd.DataFrame({
    'Pregnancies': [5],
    'Glucose': [148],
    'BloodPressure': [72],
    'SkinThickness': [35],
    'Insulin': [0],
    'BMI': [33.6],
    'DiabetesPedigreeFunction': [0.627],
    'Age': [50]
})


# In this case, we're predicting the probability of having diabetes (class 1)
predicted_probabilities = model.predict_proba(new_patient_data)
# The second column (index 1) contains the probability of class 1
probability_of_diabetes = predicted_probabilities[0][1]

# Using a threshold to make a binary prediction (0 or 1)
threshold = 0.5
predicted_class = 1 if probability_of_diabetes > threshold else 0

if predicted_class == 1:
    print("The patient is predicted to have diabetes.")
else:
    print("The patient is predicted not to have diabetes.")

# Printing the probability score
print(f"Probability of having diabetes: {probability_of_diabetes:.2f}")




#creatring a function for the above
import pandas as pd
import numpy as np

# Define a function to predict diabetes probability
def predict_diabetes_probability():
    # Get user inputs for patient data
    pregnancies = int(input("Enter the number of pregnancies: "))
    glucose = float(input("Enter the glucose level (mg/dL): "))
    blood_pressure = float(input("Enter the blood pressure (mm Hg): "))
    skin_thickness = float(input("Enter the skin thickness (mm): "))
    insulin = float(input("Enter the insulin level (μU/ml): "))
    bmi = float(input("Enter the BMI (Body Mass Index): "))
    diabetes_pedigree_function = float(input("Enter the Diabetes Pedigree Function: "))
    age = int(input("Enter the age (years): "))

    # Create a new DataFrame for the patient
    new_patient_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree_function],
        'Age': [age]
    })

    # In this case, we're predicting the probability of having diabetes (class 1)
    predicted_probabilities = model.predict_proba(new_patient_data)

    # The second column (index 1) contains the probability of class 1
    probability_of_diabetes = predicted_probabilities[0][1]

    # Using a threshold to make a binary prediction (0 or 1)
    threshold = 0.5
    predicted_class = 1 if probability_of_diabetes > threshold else 0

    if predicted_class == 1:
        print("The patient is predicted to have diabetes.")
    else:
        print("The patient is predicted not to have diabetes.")

    # Printing the probability score
    print(f"Probability of having diabetes: {probability_of_diabetes:.2f}")

# Call the function to make predictions based on user inputs
predict_diabetes_probability()
