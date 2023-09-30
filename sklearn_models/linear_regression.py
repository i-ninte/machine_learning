from sklearn import datasets
from sklearn impot linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#creating the boston variable to house the dataset

boston = data.load_boston()
X= boston.data
y= boston.target


print(X)
print(X.shape)


priny(y)
print(y.shape)

#creating the linear_regression_model
l_reg= linear_model.LinearRegression()



##visualizing the model
plt.scatter(X.T[5], y)
plt.show()

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2)


# training 
model=l_reg.fit(X_train, y_train)

#predictions
predictions = model.predict(y_test)

print("pedictions: ", predictions)
print(" R square values", l_reg.score(X,y))

print("coefficient", l_reg.coef_)
print("intercept", l_reg.intercept)
