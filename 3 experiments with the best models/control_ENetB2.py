#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pip install tensorflow_addons


# In[ ]:


import os, shutil
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


from keras.models import load_model
model = load_model('/content/drive/MyDrive/dataset/ENetB2.h5')


# In[ ]:


import pandas as pd


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


original_dataset_dir = '/content/drive/MyDrive/data/'
base_dir = '/content/drive/MyDrive/data/'

train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

batch = 10

test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=batch,
    class_mode='categorical',
    shuffle=False)


# In[ ]:


x_test = test_generator.filepaths
y_test = test_generator.labels


# In[ ]:


y_pred = model.predict(test_generator)

print(y_pred)


# In[ ]:


predicted_classes = np.argmax(y_pred, axis = 1)


# In[ ]:


print(predicted_classes)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


report = classification_report(y_test, predicted_classes)
print(report)

