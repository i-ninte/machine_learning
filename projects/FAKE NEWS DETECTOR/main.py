# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Load fake and true news datasets
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

# Create a 'class' column for labeling (0 for fake, 1 for true)
df_fake['class'] = 0
df_true['class'] = 1

# Remove the last 10 rows from both datasets for manual testing
df_fake_manual_testing = df_fake.tail(10)
df_fake.drop(df_fake.index[23470:23480], inplace=True)

df_true_manual_testing = df_true.tail(10)
df_true.drop(df_true.index[21406:21416], inplace=True)

# Concatenate fake and true datasets
df = pd.concat([df_fake, df_true], axis=0)

# Drop unnecessary columns
df = df.drop(['title', 'subject', 'date'], axis=1)

# Shuffle the DataFrame
df = df.sample(frac=1)

# Reset the index
df.reset_index(inplace=True)
df.drop(['index'], axis=1, inplace=True)

# Word processing function
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*\]', '', text)
    text = re.sub('\W', " ", text)
    text = re.sub('https?://\S+|www.\S+', '', text)
    text = re.sub('<.*?>', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Apply word processing to the 'text' column
df['text'] = df['text'].apply(wordopt)

# Split the data into features (X) and labels (y)
X = df['text']
y = df['class']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Vectorize the text using TF-IDF
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(X_train)
xv_test = vectorization.transform(X_test)

# Train a Logistic Regression model
reg = LogisticRegression()
reg.fit(xv_train, y_train)
pred_reg = reg.predict(xv_test)

# Evaluate Logistic Regression model
reg_accuracy = reg.score(xv_test, y_test)
print("Logistic Regression Accuracy:", reg_accuracy)
print("Logistic Regression Classification Report:\n", classification_report(y_test, pred_reg))

# Train a Decision Tree Classifier
dt_clf = DecisionTreeClassifier()
dt_clf.fit(xv_train, y_train)
pred_dt = dt_clf.predict(xv_test)

# Evaluate Decision Tree Classifier
dt_accuracy = dt_clf.score(xv_test, y_test)
print("Decision Tree Classifier Accuracy:", dt_accuracy)
print("Decision Tree Classifier Classification Report:\n", classification_report(y_test, pred_dt))

# Train a Gradient Boosting Classifier
gb = GradientBoostingClassifier()
gb.fit(xv_train, y_train)
gb_pred = gb.predict(xv_test)

# Evaluate Gradient Boosting Classifier
gb_accuracy = gb.score(xv_test, y_test)
print("Gradient Boosting Classifier Accuracy:", gb_accuracy)
print("Gradient Boosting Classifier Classification Report:\n", classification_report(y_test, gb_pred))

# The Decision Tree Classifier is chosen as the best model for this task

# Function to output label based on the prediction
def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "True News"

# Function for manual testing of a news input
def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_dt = dt_clf.predict(new_xv_test)
    gb_pred = gb.predict(new_xv_test)
    print("\n\nDecision Tree Prediction: " + output_label(pred_dt[0]) + ", Gradient Boost Prediction: " + output_label(gb_pred[0]))

# Input for manual testing
news_input = str(input("Enter news for manual testing: "))
manual_testing(news_input)

