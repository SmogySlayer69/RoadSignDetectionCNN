import keras,os
import numpy
import tensorflow as tf
import xml.etree.ElementTree as et

from PIL import Image
from keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPool2D , Dropout, Flatten
from keras.metrics import MeanIoU
from tensorflow.python.keras import layers
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam 


from sklearn.model_selection import train_test_split

w, h = 128, 128

X = numpy.empty((875, w, h, 4))
Y = numpy.empty((875, 1, 1, 1, 1))

X_train = numpy.empty((599, w, h, 4))
Y_train = numpy.empty((599, 1, 1, 1, 1))

X_test = numpy.empty((276, w, h, 4))
Y_test = numpy.empty((276, 1, 1, 1, 1))

for a in range(875):
    imgPath = 'archive/images/road' + str(a) + '.png'

    image = Image.open(imgPath)
    image = image.resize((w, h))
    np_img = numpy.array(image)
  
    X[a] = np_img

    antPath = 'archive/annotations/road' + str(a) + '.xml'
    y = et.parse(antPath)
    anno_file = y.getroot()

    x1 = float(anno_file[4][5][0].text)/w
    y1 = float(anno_file[4][5][1].text)/h
    x2 = float(anno_file[4][5][2].text)/w
    y2 = float(anno_file[4][5][3].text)/h

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=600, random_state=42)



input_shape = (w, h, 4)
input = tf.keras.layers.Input(input_shape)

base = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', )(input)
base = tf.keras.layers.MaxPooling2D()(base)

base = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(base)
base = tf.keras.layers.MaxPooling2D()(base)

base = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(base)
base = tf.keras.layers.MaxPooling2D()(base)
base = tf.keras.layers.Flatten()(base)

id = tf.keras.layers.Dense(64, activation='relu')(base)
id = tf.keras.layers.Dense(32, activation='relu')(id)
id = tf.keras.layers.Dense(4, activation='sigmoid')(id)

model = tf.keras.Model(input, outputs=[id])

model.compile(loss=tf.keras.losses.mse, optimizer='sgd', metrics=['accuracy'])

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=50, epochs=10)
#Accuracy does not change w/ changing of model layers, batch size, or epochs 
