import numpy as np
cvalues = [20.1, 20.8, 21.9, 22.5, 22.7, 22.3, 21.8, 21.2, 20.9, 20.1]

c=np.array(cvalues)


def fahrenheit(num):
	
	return c*9/5+ 32	

print(fahrenheit(c))

print(type(c))


import matplotlib.pyplot as plt
#print(plt.plot(c))
#print(plt.show())


# calculating the size of memory
from sys import getsizeof as size
lst= [24,12,57]
size_of_list_object= size(lst)
size_of_elements= len(lst) * size(lst[0])
total=size_of_list_object + size_of_elements

print("size without the size of the elements: ", size_of_list_object)
print("size of all the elements: :" , size_of_elements)
print("size of list, including elements", total)




a= np.array([24,12,57])
print("the memory allocation for array a is ",size(a))


# calculating the memory allocation for an empty array
e= np.array([])
print("the memory allocation for an empty array is ",size(e))
