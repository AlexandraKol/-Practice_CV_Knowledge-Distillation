#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import os, shutil
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


original_dataset_dir = '/content/drive/MyDrive/data binary'
base_dir = '/content/drive/MyDrive/data binary'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')


# In[ ]:


conv_base = tf.keras.applications.EfficientNetB0(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))


# In[ ]:


conv_base.summary()


# In[ ]:


from keras import models
from keras import layers

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# In[ ]:


model.summary()


# In[ ]:


print('This is the number of trainable weights '
      'before freezing the conv base:', len(model.trainable_weights))


# In[ ]:


conv_base.trainable = False


# In[ ]:



print('This is the number of trainable weights '
      'after freezing the conv base:', len(model.trainable_weights))


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history = model.fit_generator(\n      train_generator,\n      steps_per_epoch=100,\n      epochs=5,\n      validation_data=validation_generator,\n      validation_steps=50,\n      verbose=2)')


# In[ ]:


model.save('/content/drive/MyDrive/data binary/binary_2.h5')


# In[ ]:


import pandas as pd


# In[ ]:


test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        shuffle=False,
        class_mode='binary')


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

