
from importlib import reload

from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization, Conv2DTranspose


import model
reload(model)
from model.unet_model import get_unet_64, get_unet_128
from utils.timeit_decorator import timeit

callbacks = [EarlyStopping(monitor='val_loss',
                           patience=8,
                           verbose=1,
                           min_delta=1e-4)
             ,ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=4,
                               verbose=1,
                               epsilon=1e-4)
            ,ModelCheckpoint(monitor='val_loss',
                            filepath='./weights/best_weights.hdf5',
                            save_best_only=True,
                            save_weights_only=True)
            ,TensorBoard(log_dir='logs')
            ]


model = get_unet_64(input_shape=(128, 128, 3),num_classes=1,filters=64)
model.summary()
batch_size,nb_epoch = 4,100

#@timeit
model.fit(x=X_train_org, y=Y_train_mask.reshape(len(Y_train_mask),128, 128, 1), batch_size=batch_size, epochs=nb_epoch,
              shuffle=True, verbose=2, validation_split=0.2
              ,callbacks=callbacks
          )

Y_train_mask.shape