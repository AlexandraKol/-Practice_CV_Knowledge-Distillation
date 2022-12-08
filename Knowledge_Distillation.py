import tensorflow as tf
tf.random.set_seed(666)

from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras import models
from tensorflow.keras import layers
from keras.models import load_model
from tensorflow.keras import optimizers
from tensorflow import keras
import os, shutil
import numpy as np
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
import tensorflow_addons as tfa

data_path = 'D:/ЧелГУ/Практика/датасеты/data/'

train_dir = data_path + 'train'
val_dir = data_path + 'val'
test_dir = data_path + 'test'

img_height = 224
img_width = 224
img_depth = 3
img_size = (img_height, img_width)
batch_size = 50
EPOCHS = 15
AUTO = tf.data.experimental.AUTOTUNE

train_dataset = (
    tf.keras.utils.image_dataset_from_directory(
                  train_dir,
                  seed=123,
                  image_size=(img_height, img_width),
                  batch_size=batch_size)
                  .shuffle(100)
                  .prefetch(AUTO))

val_dataset = (
    tf.keras.utils.image_dataset_from_directory(
                  val_dir,
                  seed=123,
                  image_size=(img_height, img_width),
                  batch_size=batch_size)
                  .shuffle(100)
                  .prefetch(AUTO))

test_dataset = (
    tf.keras.utils.image_dataset_from_directory(
        test_dir,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size))

from keras.preprocessing.image import ImageDataGenerator

batch = 50

train_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch,
    class_mode='categorical',
    shuffle=True)
label_map = (train_generator.class_indices)

val_generator = test_datagen.flow_from_directory(
        val_dir,
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

print(label_map)

# Teacher model utility
base_model = load_model('D:/ЧелГУ/Практика/ENetB4new25.h5')


def get_teacher_model():
    inputs = layers.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    classifier = models.Model(inputs=inputs, outputs=x)
    
    return classifier

get_teacher_model().summary()

# Train the teacher model
teacher_model = get_teacher_model()
teacher_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=1e-4), metrics=["accuracy"])

teacher_model.evaluate(test_generator)
teacher_model.save_weights("D:/ЧелГУ/Практика/teacher_model1.h5")
teacher_model.save_weights("teacher_model1.h5")


class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = student
        self.student = teacher

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout,Flatten,                                    Dense, Activation, GlobalAveragePooling2D, BatchNormalization,                                    AveragePooling2D, Concatenate, Activation
from tensorflow.keras.applications import mobilenet, densenet
from tensorflow.keras.models import load_model, Model, clone_model

student_model = keras.Sequential(
    [
        keras.Input(shape=(224, 224, 3)),
        layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
        layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same"),
        layers.Flatten(),
        layers.Dense(8),
    ],
    name="student",
)
# Clone student for later comparison
student_scratch = keras.models.clone_model(student_model)

# Initialize and compile distiller
distiller = Distiller(student=student_model, teacher=teacher_model)
distiller.compile(
    optimizer=optimizers.RMSprop(learning_rate=1e-4),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
    student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=10,
)

# Distill teacher to student
distiller.fit(train_dataset, epochs=3)

# Evaluate student on test dataset
distiller.evaluate(test_dataset)

student_scratch.compile(
    optimizer=optimizers.RMSprop(learning_rate=1e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# Train and evaluate student trained from scratch.
student_scratch.fit(train_dataset, epochs=15)

student_scratch.evaluate(test_dataset)

student_scratch.save_weights("D:/ЧелГУ/Практика/student_model1.h5")
student_scratch.save_weights("student_model1.h5")

# Investigate the sizes
get_ipython().system('ls -lh *.h5')
