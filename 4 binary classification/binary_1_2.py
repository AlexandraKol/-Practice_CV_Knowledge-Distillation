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
                  input_shape=(256, 256, 3))


# In[ ]:


conv_base.summary()


# In[ ]:


get_ipython().run_cell_magic('time', '', "datagen = ImageDataGenerator(rescale=1./255)\nbatch_size = 20\n\ndef extract_features(directory, sample_count):\n    features = np.zeros(shape=(sample_count, 8, 8, 1280))\n    labels = np.zeros(shape=(sample_count))\n    generator = datagen.flow_from_directory(\n        directory,\n        target_size=(256, 256),\n        batch_size=batch_size,\n        class_mode='binary')\n    i = 0\n    for inputs_batch, labels_batch in generator:\n        features_batch = conv_base.predict(inputs_batch)\n        features[i * batch_size : (i + 1) * batch_size] = features_batch\n        labels[i * batch_size : (i + 1) * batch_size] = labels_batch\n        i += 1\n        if i * batch_size >= sample_count:\n\n            break\n    return features, labels\n\ntrain_features, train_labels = extract_features(train_dir, 4558)\nvalidation_features, validation_labels = extract_features(validation_dir, 1302)\ntest_features, test_labels = extract_features(test_dir, 652)")


# In[ ]:


train_features = np.reshape(train_features, (4558, 8 * 8 * 1280))
validation_features = np.reshape(validation_features, (1302, 8 * 8 * 1280))
test_features = np.reshape(test_features, (652, 8 * 8 * 1280))


# In[ ]:


get_ipython().run_cell_magic('time', '', "from keras import models\nfrom keras import layers\nfrom keras import optimizers\n\nmodel = models.Sequential()\nmodel.add(layers.Dense(256, activation='relu', input_dim=8 * 8 * 1280))\nmodel.add(layers.Dropout(0.5))\nmodel.add(layers.Dense(1, activation='sigmoid'))\n\nmodel.compile(optimizer=optimizers.RMSprop(lr=2e-5),\n              loss='binary_crossentropy',\n              metrics=['acc'])\n\nhistory = model.fit(train_features, train_labels,\n                    epochs=30,\n                    batch_size=20,\n                    validation_data=(validation_features, validation_labels))")


# In[ ]:


model.save('/content/drive/MyDrive/data binary/binary_1().h5')


# In[ ]:


pred = model.predict(test_features)


# In[ ]:


predicted_classes = np.argmax(pred, axis = 1)
print(predicted_classes)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


report = classification_report(test_labels, predicted_classes)
print(report)

