import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.EfficientNetB7 import preprocess_input
from tensorflow.keras.metrics import Precision, Recall

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    print(e)

model = EfficientNetB7(include_top=False, weights="imagenet")
global_average_layer = GlobalAveragePooling2D()(model.layers[-1].output)
output = Dense(10, activation='softmax')(global_average_layer)
model = Model(inputs=model.inputs, outputs=output)
model.summary()

datagen=ImageDataGenerator(preprocessing_function=preprocess_input, 
                           horizontal_flip=True, 
                           shear_range=5,
                           zoom_range=0.1,
                           width_shift_range=0.1,
                           height_shift_range=0.1,
                           fill_mode='nearest',
                           rotation_range=5,
                          )

train_it = datagen.flow_from_directory('original_imagenet_images/train/',
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

val_it = datagen.flow_from_directory('original_imagenet_images/val/',
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=False)

lr_schedule = ExponentialCyclicalLearningRate(
            initial_learning_rate=1e-4,
            maximal_learning_rate=1e-2,
            step_size=2000,
            scale_mode="cycle",
            gamma=0.96,
            name="MyCyclicScheduler")



optim = Adam(learning_rate=lr_schedule)

model.compile(optimizer=optim,loss='categorical_crossentropy',metrics=['accuracy', Precision(), Recall()])

callback = ModelCheckpoint(filepath='models/efficientnetb7_{epoch:03d}.hdf5', verbose=2, period=1, save_weights_only=False, save_best_only=True)

model.fit_generator(train_it, steps_per_epoch=len(train_it), validation_data=val_it, validation_steps=len(val_it), epochs=200, verbose=1, callbacks=[callback])
