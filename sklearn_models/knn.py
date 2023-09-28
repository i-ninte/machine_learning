import numpy  as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv("car.data")
print(data.head())


# creating labels and features 
#features
X= data[['buying','maint','safety' ]].values

# labels
y=data[['class']]

print(X,y)

# the strings will have to be converted to numbers hence the need for the LabelEncoder
Le= LabelEncoder()
for i  in range(len(X[0])):
 	X[:,i] = Le.fit_transform(X[:,i])
	
print(X)

#using a map to convert y 
label_mapping= {
    'unacc': 0,
    'acc':1,
    'good':2,
    'vgood':3
                
}

y['class']= y['class'].map(label_mapping)
# converting y to a numpy array 
y= np.array(y)
print(y)


#creating model
knn= neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')

#separating the model into training and testingn data 
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size= 0.2)

#training the model(fitiing )

knn.fit(X_train, y_train)

# making predictions 
prediction = knn.predict(X_test)

# checking the performance of the knn model
accuracy = metrics.accuracy_score(y_test, prediction)

print("predictions", prediction)
print("accuracy", accuracy)


## Testing the model
print("Actual value:", y[20])  # Accessing actual value by index
print("Predicted value:", knn.predict(X)[20])  # Accessing predicted value by index

