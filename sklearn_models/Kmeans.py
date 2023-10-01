from sklearn.datasets import lod_breast_cancer
from sklearn.cluster import Kmeans
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import pandaas as pd

bc=load_breast_cancer()
print(bc)

X=bc.data
#scaling the model
X=scale(bc.data)
print (X)

y= bc.target


X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.2)

model= Kmeans(n_clusters=2, random_state=0)

model.fit(X_train)


predictions= model.pedict(x_test)

labels=model.labels
print('labels',labels)
print('predictions', predictions)
print('accuracy', accuracy_score(y_test, predictions))
print('actual', y_test)

#cross tabulation test
print(pd.crosstab(y_train, labels))


#sklean checking the metrics 
def bench_k_means(estimator, name, data):
	estimator.fit(data)
	print()
