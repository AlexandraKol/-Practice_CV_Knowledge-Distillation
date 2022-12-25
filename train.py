import tensorflow as tf
import tensorflow_addons as tfa
import os
import shutil
import numpy as np
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers

original_dataset_dir = 'D:/ЧелГУ/Практика/датасеты/data'
base_dir = 'D:/ЧелГУ/Практика/датасеты/data'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

conv_base = tf.keras.applications.EfficientNetB4(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3))


model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(8, activation='softmax'))

model.summary()

conv_base.trainable = False

batch = 50

train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch,
    class_mode='categorical',
    shuffle=True)

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(256, 256),
        batch_size=batch,
        class_mode='categorical',
        shuffle=True)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=batch,
    class_mode='categorical',
    shuffle=False)
label_map = (train_generator.class_indices)
print(label_map)

train_stop = train_generator.samples//batch
validation_stop = validation_generator.samples//batch
test_stop = test_generator.samples//batch

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=1e-4),
              metrics=['acc'])

history = model.fit(train_generator,
                    steps_per_epoch=train_stop,
                    epochs=25)

test = model.evaluate(test_generator, steps=test_stop)

model.save('D:/ЧелГУ/Практика/ENetB4new25.h5')
