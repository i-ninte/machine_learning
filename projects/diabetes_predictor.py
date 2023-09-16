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

# Print the result
if predicted_class == 1:
    print("The patient is predicted to have diabetes.")
else:
    print("The patient is predicted not to have diabetes.")

# You can also print the probability score
print(f"Probability of having diabetes: {probability_of_diabetes:.2f}")
