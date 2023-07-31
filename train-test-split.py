from tensorflow.python.keras.utils import np_utils
from PIL import Image

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten

import numpy
import xml.etree.ElementTree as et
from sklearn.model_selection import train_test_split

w, h = 128, 128

X = numpy.empty((875, w, h, 4))
Y = numpy.empty((875))

X_train = numpy.empty((599, w, h, 4))
Y_train = numpy.empty((599))

X_test = numpy.empty((276, w, h, 4))
Y_test = numpy.empty((276))
#print(Y_train)

for a in range(875):
  imgPath = 'archive/images/road' + str(a) + '.png'

  image = Image.open(imgPath)
  image = image.resize((w, h))
  np_img = numpy.array(image)
  
  # np_img = np_img[0:166, 0:254, 0:4]

  X[a] = np_img

  antPath = 'archive/annotations/road' + str(a) + '.xml'
  y = et.parse(antPath)
  anno_file = y.getroot()

  anno = anno_file[4][0].text
  # print(anno)
  
  if anno == 'trafficlight':
    Y[a] = 0

  elif anno == 'stop':
    Y[a] = 1

  elif anno == 'speedlimit':
    Y[a] = 2

  elif anno == 'crosswalk':
    Y[a] = 3
  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=600, random_state=42)

# for a in range(len(X_test)):
#   imgPath = 'archive/images/road' + str(a+600) + '.png'

#   image = Image.open(imgPath)
#   image = image.resize((w, h))
#   np_img = numpy.array(image)
  
#   # np_img = np_img[0:166, 0:254, 0:4]

#   X_test[a] = np_img

#   antPath = 'archive/annotations/road' + str(a+600) + '.xml'
#   y = et.parse(antPath)
#   anno_file = y.getroot()

#   anno = anno_file[4][0].text
#   if anno == 'trafficlight':
#     Y_test[a] = 0

#   elif anno == 'stop':
#     Y_test[a] = 1

#   elif anno == 'speedlimit':
#     Y_test[a] = 2

#   elif anno == 'crosswalk':
#     Y_test[a] = 3


#X_train = X_train.reshape(X_train[0], 166, 254, 4)
#X_test = X_test.reshape(X_test[0], 166, 254, 4)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

n_classes = 4
print("Shape before one-hot encoding: ", Y_train.shape)
Y_train = np_utils.to_categorical(Y_train, n_classes)
Y_test = np_utils.to_categorical(Y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)



model = Sequential()

# model.add(Conv2D(50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(w, h, 4)))


# model.add(Conv2D(75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Dropout(0.3))


# model.add(Conv2D(125, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Dropout(0.3))

# model.add(Conv2D(250, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(500, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(250, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(4, activation='softmax'))


model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
#print(Y_test)

model.fit(X_train, Y_train, batch_size=1, epochs=15, validation_data=(X_test, Y_test))
