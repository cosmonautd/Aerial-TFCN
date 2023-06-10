import os
import warnings
import collections

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from contextlib import redirect_stderr

with redirect_stderr(open(os.devnull, "w")):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv2D,
        MaxPooling2D,
        Dropout,
        Conv2DTranspose,
        UpSampling2D,
    )
    from tensorflow.keras.callbacks import ModelCheckpoint

    from tensorflow.keras.layers import BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from keras import regularizers

Data = collections.namedtuple("Data", "x_train y_train x_val y_val x_test y_test")


class TFCN:
    def __init__(self, dims=(1000, 1000, 3)):
        self.__model__ = self.__tfcn__(dims)
        self.history = None

    def __tfcn__(self, dims):
        weight_decay = 1e-3
        M = Sequential()
        M.add(
            Conv2D(
                8,
                kernel_size=3,
                padding="same",
                activation="relu",
                kernel_regularizer=regularizers.l2(weight_decay),
                input_shape=dims,
            )
        )
        M.add(BatchNormalization())
        M.add(MaxPooling2D(pool_size=2))
        M.add(Dropout(0.2))

        M.add(
            Conv2D(
                32,
                kernel_size=3,
                padding="same",
                activation="relu",
                kernel_regularizer=regularizers.l2(weight_decay),
            )
        )
        M.add(BatchNormalization())
        M.add(MaxPooling2D(pool_size=2))
        M.add(Dropout(0.3))

        M.add(
            Conv2D(
                64,
                kernel_size=3,
                padding="same",
                activation="relu",
                kernel_regularizer=regularizers.l2(weight_decay),
            )
        )
        M.add(BatchNormalization())
        M.add(MaxPooling2D(pool_size=2))

        M.add(
            Conv2DTranspose(
                32,
                kernel_size=3,
                padding="same",
                activation="relu",
                kernel_regularizer=regularizers.l2(weight_decay),
            )
        )
        M.add(UpSampling2D(size=(2, 2), interpolation="bilinear"))

        M.add(
            Conv2DTranspose(
                8,
                kernel_size=3,
                padding="same",
                activation="relu",
                kernel_regularizer=regularizers.l2(weight_decay),
            )
        )
        M.add(UpSampling2D(size=(2, 2), interpolation="bilinear"))

        M.add(
            Conv2DTranspose(
                1,
                kernel_size=3,
                padding="same",
                activation="relu",
                kernel_regularizer=regularizers.l2(weight_decay),
            )
        )
        M.add(UpSampling2D(size=(2, 2), interpolation="bilinear"))

        M.compile(loss="mean_squared_error", optimizer="adam")

        return M

    def train(self, data, wp="weights.hdf5"):
        datagen_args = dict(
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode="reflect",
            rotation_range=360,
            zoom_range=0.2,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=30,
        )

        datagen_vx = ImageDataGenerator(**datagen_args)
        datagen_vy = ImageDataGenerator(**datagen_args)
        datagen_tx = ImageDataGenerator(**datagen_args)
        datagen_ty = ImageDataGenerator(**datagen_args)

        datagen_seed = 1
        datagen_vx.fit(data.x_val, augment=True, seed=datagen_seed)
        datagen_vy.fit(data.y_val, augment=True, seed=datagen_seed)
        datagen_tx.fit(data.x_train, augment=True, seed=datagen_seed)
        datagen_ty.fit(data.y_train, augment=True, seed=datagen_seed)

        tx = datagen_tx.flow(data.x_train, seed=datagen_seed, batch_size=16)
        ty = datagen_ty.flow(data.y_train, seed=datagen_seed, batch_size=16)
        vx = datagen_vx.flow(data.x_val, seed=datagen_seed, batch_size=16)
        vy = datagen_vy.flow(data.y_val, seed=datagen_seed, batch_size=16)

        checkpoint = ModelCheckpoint(wp, verbose=1, save_best_only=True)
        callbacks = [checkpoint, EarlyStopping(min_delta=0.0001, patience=20)]

        tgen = (pair for pair in zip(tx, ty))
        vgen = (pair for pair in zip(vx, vy))

        self.history = self.__model__.fit(
            x=tgen,
            epochs=100,
            steps_per_epoch=40,
            callbacks=callbacks,
            validation_data=vgen,
            validation_steps=8,
        )

    def load(self, wp):
        self.__model__.load_weights(wp)

    def predict(self, X):
        return self.__model__(X)

    def loss(self, X, Y):
        return self.__model__.evaluate(X, Y, verbose=0)
