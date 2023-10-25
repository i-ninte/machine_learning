##HAND WRITING DIGIT RECOGNIZER
# install pillow, mnist, numpy sklearn
#using !pip install pillow mnist numpy sklearn


from PIL import Image
import numpy as np
from sklearn.neuron_network import NLPClassifier
from sklearn.matrix import confusion_matix
import mnist

#training variables 
x_train = mnist.train_images()
y_train= mnist.labels()


x_test = mnist.train_images()
y_test= mnist.labels()


print("x_train",x_train)
print("x_test",x_test)


print("y_train",y_train)

# checking the number of enteries 
print(x_train.shape)


#checking the dimension of the training data 
print(x_train.ndim)

#changing the number of enteries 
x_train=x_train.reshape((-1, 28*28))
x_test= x_test.reshape((-1, 28 *28))
# because the image is a 784, 28 * 28 pixel image

#checking the first element in our model we realized the numbers ranged from  0 to 255 but then the with neural networks the best possible range of values is -1 to 1. hence the need to optimize the model to fit this range

print(x_train[0])
#scaling the training data 
x_train=(x_train/256)
x_test= (x__test/256)

print(x_train[0])


clf= NLPClassifier(solver='adam', activation='relu', hidden_layer_sizes=(64,64)
						 
#training 
clf.fit(x_train,y_train)
predictions= clf.predict(x_test)

accuracy= confusion_matrix(y_test, predictions)
print(accuracy)

						 
						 
#the accuracy works by finding the trace of the matrix and dividing it by the
#the visual elements 
def acc(confusion_matrix):
	diagonal= confusion_matrix.trace()
	elements= confusion_matrix.sum()
	return diagonal/elements

print(acc(accuracy))		

						 
						 
						 
#next we download gimp from google and create a 28 * 28	image 
# a python program that gives the pixel and we give that pixel to our model and hopefully it returns the number we wrote with gimp
#image to bytes 
from PIL import Image 
img = Image.open("name_of_image")
data= list(img.getdata())
print(data)

						 
# after viewing our data we found out that it is the opposite of the data we had in our dataset. ous started with all 0s but this starts with all 1s
		
						 
#converting the data to suit ours
for i in range(len(data)):
	data[i]= 255- data[i]
print(data)
						 
#after running the program we copy the data and store it ina variable called five
						 
five=np.array(five)/256
#divifing by 256 to convert them within the data range
p= clf.predict([five])
print(p)						 
