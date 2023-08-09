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

    x1 = int(anno_file[4][5][0].text)/w
    y1 = int(anno_file[4][5][1].text)/h
    x2 = int(anno_file[4][5][2].text)/w
    y2 = int(anno_file[4][5][3].text)/h

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=600, random_state=42)


