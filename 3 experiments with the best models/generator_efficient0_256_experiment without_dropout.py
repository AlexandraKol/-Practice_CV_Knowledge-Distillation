#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install tensorflow_addons


# In[ ]:


import tensorflow as tf
import tensorflow_addons as tfa
import os, shutil
import numpy as np


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


original_dataset_dir = '/content/drive/MyDrive/data'
base_dir = '/content/drive/MyDrive/data'

train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')


# In[ ]:


conv_base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(256, 256, 3))


# In[ ]:


conv_base.summary()


# In[ ]:


from keras import layers
from keras import models

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(8, activation='softmax'))


# In[ ]:


model.summary()


# In[ ]:


print('Это количество тренируемых весов '
            'перед замораживанием conv base:', len(model.trainable_weights))


# In[ ]:


conv_base.trainable = False


# In[ ]:


print('Это количество тренируемых весов '
            'после заморозки conv base:', len(model.trainable_weights))


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator

batch = 10

train_datagen = ImageDataGenerator(rescale = 1. / 255, validation_split = 0.15)

test_datagen = ImageDataGenerator(rescale = 1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=batch,
    class_mode='categorical',
    shuffle=False)
label_map = (train_generator.class_indices)

validation_generator = train_datagen.flow_from_directory(
   train_dir,
   target_size = (256, 256),
   batch_size = batch,
   class_mode = 'categorical', 
   subset = "validation", 
   shuffle = False)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=batch,
    class_mode='categorical',
    shuffle=False)

label_map


# In[ ]:


train_stop = 23219//batch
validation_stop = 3479//batch
test_stop = 2585//batch


# In[ ]:


from tensorflow.keras import optimizers

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=1e-4),
              metrics=[tf.keras.metrics.CategoricalAccuracy()])


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history = model.fit(train_generator,\n                    steps_per_epoch=train_stop,\n                    epochs=10,\n                    validation_data = validation_generator,\n                    validation_steps = validation_stop,\n                    verbose = 1)')


# In[ ]:


test = model.evaluate(test_generator, steps=test_stop)


# In[ ]:


model.save('/content/drive/MyDrive/dataset/ENetB0notS.h5')

