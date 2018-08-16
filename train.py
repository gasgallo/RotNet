from __future__ import print_function

import os
import sys
from sklearn.model_selection import train_test_split

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import SGD

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import RotNetDataGenerator

path_r = '.'
data_path = os.path.join(path_r, 'data', 'images')
image_paths = []
for filename in os.listdir(data_path):
	if filename!='.dir3_0.wmd':
		image_paths.append(os.path.join(data_path, filename))

train_filenames, test_filenames = train_test_split(image_paths, test_size=0.3, random_state=42)

print(len(train_filenames), 'train samples')
print(len(test_filenames), 'test samples')

model_name = 'rotnet_resnet50_full'

# number of classes
nb_classes = 4
# input image shape
input_shape = (224, 224, 3)
# path to the weights
weights_path = os.path.join(path_r, 'data', 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
#weights_path = os.path.join(path_r, 'data', 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

# load base model
base_model = ResNet50(weights=weights_path, include_top=False,
                      input_shape=input_shape)
#base_model = VGG16(weights=weights_path, include_top=False,
#                      input_shape=input_shape)

# create the new model
model = Sequential()
model.add(base_model)
model.add(Dropout(0.5))
model.add(Flatten())
#model.add(Dense(256, activation='relu', name='fc256'))
model.add(Dense(nb_classes, activation='softmax', name='fc4'))
model.layers[0].trainable = False 

model.summary()

# model compilation
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.01, momentum=0.9),
              metrics=['accuracy'])

# training parameters
batch_size = 64
nb_epoch = 50

output_folder = 'models'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# callbacks
monitor = 'val_acc'
checkpointer = ModelCheckpoint(
    filepath=os.path.join(output_folder, model_name + '.hdf5'),
    monitor=monitor,
    save_best_only=True,
    verbose=1)
reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=5, verbose=1)
early_stopping = EarlyStopping(monitor=monitor, patience=10, verbose=1)
tensorboard = TensorBoard()

# training loop
model.fit_generator(
    RotNetDataGenerator(
        train_filenames,
        input_shape=input_shape,
        batch_size=batch_size,
        preprocess_func=preprocess_input,
        crop_center=True,
        crop_largest_rect=True,
        shuffle=True
    ),
    steps_per_epoch=len(train_filenames) / batch_size,
    epochs=nb_epoch,
    validation_data=RotNetDataGenerator(
        test_filenames,
        input_shape=input_shape,
        batch_size=batch_size,
        preprocess_func=preprocess_input,
        crop_center=True,
        crop_largest_rect=True
    ),
    validation_steps=len(test_filenames) / batch_size,
    callbacks=[checkpointer, reduce_lr, early_stopping, tensorboard],
    workers=10
)
