#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install tensorflow_addons


# In[2]:


import tensorflow as tf
import tensorflow_addons as tfa
import os, shutil
import numpy as np


# In[3]:


from google.colab import drive
drive.mount('/content/drive')


# In[4]:


original_dataset_dir = '/content/drive/MyDrive/data'
base_dir = '/content/drive/MyDrive/data'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')


# In[5]:


conv_base = tf.keras.applications.EfficientNetB2(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3))


# In[6]:


conv_base.summary()


# In[54]:


from keras import layers
from keras import models

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(8, activation='softmax'))


# In[55]:


model.summary()


# In[56]:


conv_base.trainable = False


# In[57]:


from keras.preprocessing.image import ImageDataGenerator

batch = 100

train_datagen = ImageDataGenerator(rescale=1./255,
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch,
    class_mode='categorical',
    shuffle=True)
label_map = (train_generator.class_indices)

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=batch,
        class_mode='categorical',
        shuffle=True)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=batch,
    class_mode='categorical',
    shuffle=False)

label_map


# In[58]:


from tensorflow.keras import optimizers

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=1e-4),
              metrics=['acc'])


# In[59]:


train_stop = train_generator.samples//batch
validation_stop = validation_generator.samples//batch


# In[ ]:


from keras.callbacks import LearningRateScheduler


# In[60]:


get_ipython().run_cell_magic('time', '', 'history = model.fit(train_generator,\n                    steps_per_epoch = train_stop,\n                    epochs=15,\n                    validation_data = validation_generator,\n                    validation_steps = validation_stop)')


# In[61]:


test = model.evaluate(test_generator)


# In[62]:


model.save('/content/drive/MyDrive/data/ENetB2notS.h5')

