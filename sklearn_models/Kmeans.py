from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score, adjusted_rand_score
from sklearn.preprocessing import scale
import pandas as pd

def bench_k_means(estimator, name, data, labels):
    estimator.fit(data)
    print(f'{name}:')
    print(f'Silhouette Score: {silhouette_score(data, estimator.labels_)}')
    print(f'Adjusted Rand Index: {adjusted_rand_score(labels, estimator.labels_)}')

bc = load_breast_cancer()
X = bc.data
X = scale(bc.data)
y = bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = KMeans(n_clusters=2, random_state=0)
model.fit(X_train)
predictions = model.predict(X_test)
labels = model.labels

print('Labels:', labels)
print('Predictions:', predictions)
print('Accuracy:', accuracy_score(y_test, predictions))
print('Actual:', y_test)

# Cross-tabulation test
print(pd.crosstab(y_train, labels))

# Benchmark K-Means
bench_k_means(model, 'K-Means', X_train, y_train)
