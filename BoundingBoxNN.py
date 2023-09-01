import keras,os
import numpy as np
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

def get_iou(ground_truth, pred):
    # coordinates of the area of intersection.
    ix1 = tf.math.maximum(ground_truth[0], pred[0])
    iy1 = tf.math.maximum(ground_truth[1], pred[1])
    ix2 = tf.math.minimum(ground_truth[2], pred[2])
    iy2 = tf.math.minimum(ground_truth[3], pred[3])
     
    # Intersection height and width.
    i_height = tf.math.maximum(iy2 - iy1 + 1, 0.)
    i_width = tf.math.maximum(ix2 - ix1 + 1, 0.)
     
    area_of_intersection = i_height * i_width
     
    # Ground Truth dimensions.
    gt_height = ground_truth[3] - ground_truth[1] + 1
    gt_width = ground_truth[2] - ground_truth[0] + 1
     
    # Prediction dimensions.
    pd_height = pred[3] - pred[1] + 1
    pd_width = pred[2] - pred[0] + 1
     
    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection
     
    iou = area_of_intersection / area_of_union
     
    return iou

w, h = 256, 256

X = np.empty((875, w, h, 4))
Y = np.empty((875, 4))

X_train = np.empty((599, w, h, 4))
Y_train = np.empty((599, 4))

X_test = np.empty((276, w, h, 4))
Y_test = np.empty((276, 4))

for a in range(875):
    imgPath = 'archive/images/road' + str(a) + '.png'

    image = Image.open(imgPath)
    w_org, h_org = image.size
    image = image.resize((w, h))
    np_img = np.array(image)
  
    X[a] = np_img

    antPath = 'archive/annotations/road' + str(a) + '.xml'
    y = et.parse(antPath)
    anno_file = y.getroot()

    x1 = float(anno_file[4][5][0].text)/w_org
    y1 = float(anno_file[4][5][1].text)/h_org
    x2 = float(anno_file[4][5][2].text)/w_org
    y2 = float(anno_file[4][5][3].text)/h_org
    Y[a] = np.array([x1, x2, y1, y2])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=600, random_state=42)

#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255


#vgg = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(w, h, 4)))

#vgg.trainable = False
#flatten = vgg.output
#flatten = Flatten()(flatten)

#Change loss function loss='categorical_crossentropy' -> regression type loss function

#bboxHead = Dense(128, activation="relu") (flatten)
#bboxHead = Dense(64, activation="relu") (bboxHead)
#bboxHead = Dense(32, activation="relu") (bboxHead)
#bboxHead = Dense(4, activation="sigmoid") (bboxHead)
#model = Model(inputs=vgg.input, outputs=bboxHead)

model = Sequential()
input_shape = (w, h, 4)
model.add(tf.keras.layers.Input(input_shape))

model.add(tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', ))
model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(4))
#input = tf.keras.layers.Input(input_shape)



#id = tf.keras.layers.Flatten()(input)
#id = tf.keras.layers.Dense(64, activation='relu')(id)
#id = tf.keras.layers.Dense(32, activation='relu')(id)
#id = tf.keras.layers.Dense(4)(id)

#model = tf.keras.Model(input, outputs=[id])

model.compile(loss=tf.keras.losses.mse, optimizer='adam', metrics=[get_iou])

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=10, epochs=10)
#Accuracy does not change w/ changing of model layers, batch size, or epochs 
