
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os import listdir
from os.path import isfile, join
from PIL import Image
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as mcolors


# In[7]:


metadf = pd.read_csv('dataset/categories.csv', sep=',')
subcat = []
no_subcat = 0
for row in metadf.name:
    subcat.append(row)
no_subcat = len(subcat)
cat = []
for row in metadf.category:
    cat.append(row)
catid = []
for row in metadf.catId:
    catid.append(row)
no_cat = 1
act = catid[0]
categories = []
categories.append(cat[0])
for i in range(len(catid)):
    if catid[i]!=act:
        categories.append(cat[i])
        no_cat+=1
        act=catid[i]

col = []
for row in metadf.color:
    c = row.replace(" ", "").split(',')
    rgb = []
    for i in c:
        rgb.append(int(i))
    col.append(rgb)


# In[8]:


import natsort

data_filenames = []
for root, dirs, files in os.walk('dataset/raw_images/'):  
    for filename in files:
        data_filenames.append(filename)

annot_filenames = []
for root, dirs, files in os.walk('dataset/class_color/'):  
    for filename in files:
        annot_filenames.append(filename)
        
data_filenames = natsort.natsorted(data_filenames)
annot_filenames = natsort.natsorted(annot_filenames)


# In[9]:


catid_annot_filenames = []
for root, dirs, files in os.walk('dataset/catid_annot/'):  
    for filename in files:
        catid_annot_filenames.append('dataset/catid_annot/'+filename)


# In[10]:


nb_samples=len(data_filenames)
valid_split = 0.15
test_split = 0.15
train_split = 0.7

data_train = np.array(data_filenames[0:int(nb_samples*(1-valid_split-test_split))])
data_valid = data_filenames[int(nb_samples*(1-valid_split-test_split)):int(nb_samples*(1-test_split))]
data_test  = data_filenames[int(nb_samples*(1-test_split)):]


# In[12]:


import tensorflow as tf
from tensorflow.python.client import device_lib
import cv2
import imageio
import json
from keras.backend.tensorflow_backend import set_session
from keras.utils.np_utils import to_categorical
from keras.applications import imagenet_utils


# In[13]:


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


# In[14]:


def preprocess_input(x):
    return imagenet_utils.preprocess_input(x, mode='tf')


# In[15]:


catid_annot_img = np.array(Image.open('dataset/catid_annot/'+ data_train[1][:-3] + "png"),dtype=np.int64)


# In[16]:


def data_generator(filenames, batch_size=32, dim=(720, 1280), n_classes=41, shuffle=True):
    # Initialization
    data_size = len(filenames)
    nbatches = data_size // batch_size
    list_IDs = np.arange(data_size)
    indices = list_IDs
    # Data generation
    while True:
        try:
            if shuffle == True:
                np.random.shuffle(indices) #shuffling when Shuffle parameter is True

            for index in range(nbatches):
                batch_indices = indices[index*batch_size:(index+1)*batch_size]

                X = np.empty((batch_size, *dim, 3))
                y_semseg = np.empty((batch_size, *dim), dtype=int)

                for i, ID in enumerate(batch_indices):
                    #reading in the raw image on the fly
                    image = cv2.resize(np.array(Image.open('dataset/raw_images/' + filenames[ID]), dtype=np.uint8), dim[1::-1])
                    #loading in the serialized annotation file on the fly
                    catid_annot_img = np.array(Image.open('dataset/catid_annot/'+ filenames[ID][:-3] + "png"),dtype=np.int64)
                    catid_annot_img = np.reshape(catid_annot_img, (720, 1280))
                    label = cv2.resize(catid_annot_img, dim[1::-1], interpolation=cv2.INTER_NEAREST)

                    X[i,] = image
                    y_semseg[i] = label

                yield (preprocess_input(X), to_categorical(y_semseg, num_classes=n_classes))
        except StopIteration as e:
            print(e)
            break


# In[17]:


#Parameters for the data generator
batch_size = 2
data_shape= np.array(Image.open('dataset/raw_images/' + data_train[0])).shape[:2]
data_shape= (int(data_shape[0]/2), int(data_shape[1]/2))
classes = no_subcat


# In[18]:


test_generator = data_generator(data_test, batch_size=batch_size, dim=data_shape, n_classes=classes)


# In[19]:


import keras.models as models
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K


# In[20]:


import keras
from keras import optimizers

def get_unet_128_ulite(input_shape=(360, 640, 3),
                 num_classes=41, optimizer='adam'):
    inputs = Input(shape=input_shape)
    # 128
 
    down1 = Conv2D(64, (3, 3), padding='same')(inputs)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64
 
    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32
 
    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
 
    # 16
 
    # 16
 
    up3 = Conv2D(256, (3, 3), padding='same')(down3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32
 
    up2 = UpSampling2D((2, 2))(up3)
    up2 = keras.layers.concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64
 
    up1 = UpSampling2D((2, 2))(up2)
    up1 = keras.layers.concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128
 
    classify = Conv2D(num_classes, (1, 1), padding='valid')(up1)
    classify = Activation('softmax')(classify)
 
    model = Model(inputs=inputs, outputs=classify)
    model.compile(loss='categorical_crossentropy',
                  optimizer= optimizer,
                  metrics=['accuracy'])
 
    return model


# In[21]:


semseg_model = get_unet_128_ulite()


# In[22]:


semseg_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
semseg_model.load_weights('model_weights/model.36-0.9417.hdf5')


# In[23]:


i = 0
image, label = next(test_generator)
label = np.argmax(label[i], axis=-1)

pred = semseg_model.predict(image)
pred = np.argmax(pred[0], axis=-1)
fig=plt.figure(figsize=(20, 10))

cm = plt.get_cmap('gist_ncar')
image = image[i]

fig.add_subplot(1, 2, 1)
plt.imshow((image * .5 + .5) * .6 + cm(label/34.)[...,:3] * .4)
fig.add_subplot(1, 2, 2)
plt.imshow((image * .5 + .5) * .6 + cm(pred/34.)[...,:3] * .4)
plt.show()

