
from sklearn import dataset
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score

iris= dataset.load_iris() # loading dataset 


#spitting dataset into features and labels

X = iris.data
y= iris.target
print(X,y)


#creating an array for classes 
classes=['Iris Setosa','Iris Versicolour', 'Iris Virginica']


# checking the number of enteries 
print(X.shape)
pint(y.shape)

# splitting 
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.3)
# checking the dimensions of the test data

print(X_train.shape)

print(y_train.shape)

# dimensions of test
print(X_test.shape)
print(y_test.shape)

#creating svm model
model= svm.SVC()

# training the model
model.fit(X_train, y_train)
print(model)

#making predictions

prediction = model.predict(X_test)

#checking accuracy
accuracy= accuracy_score(y_test, prediction)

#printing predictions and accuracy

print("predictions :", prediction)
print("actual", y_test)
print("accuracy :", accuracy)




