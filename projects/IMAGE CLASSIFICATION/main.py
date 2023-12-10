# Importing necessary libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Loading CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

# Checking the shape of the test set
X_test.shape

# Checking the shape of the training set
X_train.shape

# Checking the shape of the training labels
y_train.shape

# Checking the shape of the test labels
y_test.shape

# Displaying the first 5 training labels
y_train[:5]

# Reshaping the training labels
y_train = y_train.reshape(-1,)

# Displaying the first 5 reshaped training labels
y_train[:5]

# Reshaping the test labels
y_test = y_test.reshape(-1,)

# Defining classes for the CIFAR-10 dataset
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Function to plot a sample image with its label
def plot_sample(X, y, index):
    plt.figure(figsize=(15, 2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])

# Example usage of the plot_sample function
plot_sample(X_train, y_train, 5)

# Example usage of the plot_sample function with a different index
plot_sample(X_train, y_train, 675)

# Normalizing the pixel values of the images
X_train = X_train / 255.0
X_test = X_test / 255

# Building a simple artificial neural network (ANN) for classification
ann = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(3000, activation='relu'),
    layers.Dense(1000, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compiling the ANN model
ann.compile(optimizer='SGD',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

# Training the ANN model
ann.fit(X_train, y_train, epochs=5)

# Importing libraries for model evaluation
from sklearn.metrics import confusion_matrix, classification_report

# Making predictions using the trained ANN model
y_pred = ann.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

# Printing classification report
print("Classification report:\n", classification_report(y_test, y_pred_classes))

# Visualizing the confusion matrix
import seaborn as sns

plt.figure(figsize=(14, 7))
sns.heatmap(y_pred, annot=True)
plt.xlabel("Prediction")
plt.title("Confusion Matrix")
plt.show()

# Building a Convolutional Neural Network (CNN) model for image classification
cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compiling the CNN model
cnn.compile(optimizer="adam",
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

# Training the CNN model
cnn.fit(X_train, y_train, epochs=10)

# Evaluating the CNN model on the test set
cnn.evaluate(X_test, y_test)

# Making predictions using the trained CNN model
y_pred = cnn.predict(X_test)
y_pred[:5]

# Converting predicted probabilities to class labels
y_classes = [np.argmax(element) for element in y_pred]
y_classes[:5]

# Displaying the first 5 true labels from the test set
y_test[:5]

# Checking predictions with sample images
plot_sample(X_test, y_test, 60)

# Checking predictions with another sample image
plot_sample(X_test, y_test, 5)

# Displaying the predicted classes for the sample images
classes[y_classes[60]]

classes[y_classes[5]]
