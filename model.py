# Import all packages
#--------------------------------------------------------------------------------------


from keras import backend as K
import tensorflow as tf

print tf.__version__

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.activations import softmax

# dimensions of our images.
img_width, img_height = 299, 299

train_data_dir = '/media/samba_share/poornima/Xray_view_detector/TRAIN' #location of training data
validation_data_dir = '/media/samba_share/poornima/Xray_view_detector/VAL' #location of validation data

# number of samples used for determining the samples_per_epoch
nb_train_samples = 80
nb_validation_samples = 20
epochs = 20
batch_size = 5

train_datagen = ImageDataGenerator(
        rescale=1./255,            # normalize pixel values to [0,1]
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

val_datagen = ImageDataGenerator(
         rescale=1./255)       # normalize pixel values to [0,1]

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = train_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

base_model = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

model_top = Sequential()
model_top.add(GlobalAveragePooling2D(input_shape=base_model.output_shape[1:], data_format=None)),
model_top.add(Dense(256, activation='relu'))
model_top.add(Dropout(0.5))
model_top.add(Dense(3, activation='softmax'))

model = Model(inputs=base_model.input, outputs=model_top(base_model.output))

model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.0), loss='categorical_crossentropy', metrics=['accuracy'])

print model_top.summary()

history = model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps= 4)

import matplotlib.pyplot as plt

print(history.history.keys())

plt.figure()
plt.plot(history.history['acc'], 'orange', label='Training accuracy')
plt.plot(history.history['val_acc'], 'blue', label='Validation accuracy')
plt.plot(history.history['loss'], 'red', label='Training loss')
plt.plot(history.history['val_loss'], 'green', label='Validation loss')
plt.legend()
plt.show()

import numpy as np
from keras.preprocessing import image
import cv2
import os
base_dir = '/media/samba_share/poornima/Xray_view_detector/TEST/AP'
case_list=os.listdir(base_dir)
for case in case_list:
    print case
    img_path='/media/samba_share/poornima/Xray_view_detector/TEST/AP/{}'.format(case) #change to location of chest x-ray
    # img_path2='/media/samba_share/data/Xray_view_detector/PNG/AP/MGHCXR_165.png'  #change to location of abd x-ray

    img = image.load_img(img_path, target_size=(img_width, img_height))

    # img2 = image.load_img(img_path2, target_size=(img_width, img_height))
    plt.imshow(img)
    plt.show()

    img = image.img_to_array(img)
    x = np.expand_dims(img, axis=0) * 1./255
    score = model.predict(x)


    if score[0][0] > 0.5:
        print 'Its AP view with {} accuracy'.format(score[0][0])
    elif score[0][1] > 0.5:
        print 'It is LL view with {} accuracy'.format(score[0][1])
    else:
        print 'Its PA view with {} accuracy'.format(score[0][2])
