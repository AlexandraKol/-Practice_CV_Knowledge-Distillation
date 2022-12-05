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


conv_base = tf.keras.applications.ResNet50(
    include_top=False,
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


model.summary()


# In[ ]:


conv_base.trainable = False


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator

batch = 10

train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch,
    class_mode='categorical',
    shuffle=True)
label_map = (train_generator.class_indices)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=batch,
    class_mode='categorical',
    shuffle=True)

label_map


# In[ ]:


train_stop = 3604//batch
test_stop = 401//batch


# In[ ]:


from tensorflow.keras import optimizers

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=1e-4),
              metrics=[tf.keras.metrics.CategoricalAccuracy(),
                       tf.keras.metrics.Recall(),
                       tf.keras.metrics.Precision(),
                       tfa.metrics.F1Score(num_classes=8),
                       tfa.metrics.FBetaScore(num_classes=8)])


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history = model.fit(train_generator,\n                    steps_per_epoch=train_stop,\n                    epochs=10)')


# In[ ]:


test = model.evaluate(test_generator, steps=test_stop)


# In[ ]:


model.save('/content/drive/MyDrive/dataset/ResNet50.h5')


# In[ ]:


import pandas as pd


# In[ ]:


df_marks = pd.DataFrame({'x_test': test_generator.filepaths,
     'y_test': test_generator.labels})

x_test = df_marks.drop(['y_test'],axis=1)
y_test = df_marks['y_test']


# In[ ]:


pred = model.predict(test_generator)


# In[ ]:


predicted_classes = np.argmax(pred, axis = 1)
print(predicted_classes)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


report = classification_report(y_test, predicted_classes)
print(report)

