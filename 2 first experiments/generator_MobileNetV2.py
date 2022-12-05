#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install tensorflow_addons


# In[ ]:


import tensorflow as tf
import tensorflow_addons as tfa
import os, shutil
import numpy as np
import pandas as pd


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


original_dataset_dir = '/content/drive/MyDrive/data/'
base_dir = '/content/drive/MyDrive/data/'

train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')


# In[ ]:


conv_base = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    input_shape=(224, 224, 3))


# In[ ]:


conv_base.summary()


# In[ ]:


from keras import layers
from keras import models

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(8, activation='softmax'))


# In[ ]:


# model.summary()


# In[ ]:


conv_base.trainable = False


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator

batch = 10
epochs = 5

train_datagen = ImageDataGenerator(
    rescale = 1. / 255, 
    validation_split = 0.15,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch,
    class_mode='categorical',
    subset = "training")
label_map = (train_generator.class_indices)

validation_generator = train_datagen.flow_from_directory(
   train_dir,
   target_size = (224, 224),
   batch_size = batch,
   class_mode = 'categorical', 
   subset = "validation", 
   shuffle = False)


label_map


# In[ ]:



test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=batch,
    class_mode='categorical',
    shuffle=True)


# In[ ]:


test = model.evaluate(test_generator, return_dict=True)


# In[ ]:


model.save('/content/drive/MyDrive/dataset/MobileNetV2.h5')


# In[ ]:


df_marks = pd.DataFrame({'x_test': test_generator.filepaths,
     'y_test': test_generator.labels})


# In[ ]:


df_marks.to_csv('/content/drive/MyDrive/dataset/file.csv')


# In[ ]:


x_test = df_marks.drop(['y_test'],axis=1)
y_test = df_marks['y_test']


# In[ ]:


pred = model.predict(test_generator)

print(pred)


# In[ ]:


predicted_classes = np.argmax(pred, axis = 1)


# In[ ]:


print(predicted_classes)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


report = classification_report(y_test, predicted_classes)
print(report)

