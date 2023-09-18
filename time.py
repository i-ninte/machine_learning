import pandas as pd
import numpy as np
import time
size_of_vec= 1000
X = np.arange(size_of_vec)
Y= np.arange(size_of_vec)
Z= X + Y
def pure_python_version():
	t1 = time.time()
	
	x = range(size_of_vec)
	y = range(size_of_vec)
	z = [X[i] + Y[i] for i in range(len(X))  ] 
	return time.time() - t1



def numpy_version():
	t1= time.time()
	
	return time.time()


t1= pure_python_version()
t2= numpy_version()


print(t1,t2)
print("Numpy is in the example" + str(t1/t2) + " faster ")
