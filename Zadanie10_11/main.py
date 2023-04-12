import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from keras.preprocessing.image import ImageDataGenerator
from keras import Input
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import backend as K
from keras import callbacks


# dimensions of our images.
img_width, img_height = 200, 200

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
log_dir_root = 'logs'
nb_train_samples = 2100
nb_validation_samples = 700
epochs = 5

# create logdir if does not exist
os.makedirs(log_dir_root, exist_ok=True)

# options
options = [
    {
        "name": "exp_000",
        "activation": "relu",
        "conv_num": 3,
        "filter_num": 32,
        "filter_size": (3, 3),
        "max_pooling": (2, 2),
        "dropout": 0.2,
        "optimizer": "adam",
        "batch_size": 16,
    },
]

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

for option in options:
    # create log direction for current experiment
    log_dir = os.path.join(log_dir_root, option["name"])
    os.makedirs(log_dir, exist_ok=True)

    # model definition
    inputs = Input(input_shape)

    x = Conv2D(option["filter_num"], option["filter_size"], activation=option["activation"], kernel_initializer='he_uniform', padding="same")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(option["max_pooling"])(x)
    x = Dropout(option["dropout"])(x)

    for i in range(1, option["conv_num"]):
        x = Conv2D(option["filter_num"] * (2 ** i), option["filter_size"], activation=option["activation"], kernel_initializer='he_uniform', padding="same")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(option["max_pooling"])(x)
        x = Dropout(option["dropout"])(x)

    x = Flatten()(x)
    x = Dense(128, activation=option["activation"], kernel_initializer='he_uniform')(x)
    x = Dropout(option["dropout"])(x)
    x = Dense(1)(x)
    outputs = Activation('sigmoid')(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(
        loss='binary_crossentropy',
        optimizer=option["optimizer"],
        metrics=['accuracy']
    )

    # print model information
    model.summary()

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # loading a generator of train inputs for neural network
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=option["batch_size"],
        class_mode='binary',
    )

    # loading a generator of validation inputs for neural network
    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=option["batch_size"],
        class_mode='binary',
    )

    # callbacks definition
    callback_list = list()

    # tensorboard callback
    tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_images=False)
    callback_list.append(tb_cb)

    # saving model at best results
    mcp_save = callbacks.ModelCheckpoint(
        os.path.join(log_dir, 'best.hdf5'),
        save_best_only=True,
        monitor='val_loss',
        mode='min',
    )
    callback_list.append(mcp_save)

    # train the model
    model.fit(
        train_generator,
        steps_per_epoch=nb_train_samples // option["batch_size"],
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // option["batch_size"],
        callbacks=callback_list,
    )
