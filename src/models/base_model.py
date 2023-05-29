import tensorflow

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras import layers, models

class Base_CNN:
    def initialize_model():

        target_shape = (512, 512, 3)
        model = models.Sequential()

        model.add(layers.Conv2D(16, (5,5), input_shape=target_shape, padding='same', activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2,2)))
        model.add(layers.Conv2D(32, (4,4), activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2,2)))
        model.add(layers.Conv2D(64, (3,3), activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2,2)))
        model.add(layers.Conv2D(64, (2,2), activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2,2)))

        model.add(layers.Flatten())

        model.add(layers.Dense(64, activation='relu'))

        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

        return model
