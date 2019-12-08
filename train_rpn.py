"""
Commmand to Run train_rpn.py - 

python train_rpn.py --input_path serengeti /
                    --dataset_name serengeti /
                    --network vgg16 /
                    --n_epochs 5 /
                    --n_epoch_length 10

"""

### INITIAL STEPS ####
"""
1. GET DATA.
2. CREATE CONFIG FILE.
3. DEFINING A DATA GENERATOR.
"""


### SETTING UP COMMAND LINE INTERFACE ####

import numpy as np
import argparse
import os

model_save_path = os.path.join("models", "rpn")

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type = str, required = True, dest = "input_path", help = "Path to Input Directory")
parser.add_argument("--dataset_name", type = str, required = True, dest = "dataset_name", help = "Name of the Dataset")
parser.add_argument("--network", type = str, required = False, default = "vgg16", dest = "network", help = "Base Network used as Backbone")
parser.add_argument("--n_epochs", type = int, required = False, default = 2, dest = "n_epochs", help = "No. of Epochs to Train the Network")
parser.add_argument("--n_epoch_length", type = int, required = False, default = 10, dest = "n_epoch_length", help = "Length of Each Epoch")
parser.add_argument("--model_save_path", type = str, required = False, default = model_save_path, dest = "model_save_path", help = "Path to Model Directory where Network Weights are stored")


args = parser.parse_args()
input_dir = args.input_path
dataset_name = args.dataset_name
base_network = args.network

from data_processing import data_parser


### Gathering details about the dataset ###

class_count, class_mapping, img_data = data_parser.get_data(input_dir, dataset_name)
print(class_count, class_mapping)


### Setting up Config File ###

from config import Config

config = Config()

config.class_count = class_count
config.class_mapping = class_mapping

### Generating Input Data and the Labels ###

"""
X = input image.
Y = classification and regression labels.

Defining a Data Generator to produce batches of Data.

1 Batch = 1 Image.
"""

### Testing the Data Generator ###

import data_generators.anchors as anchors

train_data_generator = anchors.get_anchor_gt(config, img_data)

# (X, Y) = next(train_data_generator)
# print(X, Y)


### STEPS ###
"""
1. Create Base Network.
2. Define RPN Network.
3. Define Losses.
4. Obtain Feature Maps.
5. Generate Proposals Using RPN.
"""


### Base Network ###

from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from networks import vgg16 as network
from metrics import losses


input_shape_img = (None, None, 3)
input_shape_roi = (None, 4)

img_input = Input(shape = input_shape_img)
roi_input = Input(shape = input_shape_roi)
n_anchors = len(config.anchor_box_ratios) * len(config.anchor_box_scales)

shared_features = network.nn_base(input_tensor = img_input, trainable = True)

### Region Proposal Network ###

rpn = network.rpn(base_layers = shared_features, num_anchors = n_anchors)
model_rpn = Model(img_input, rpn[:2])

### Loading Weights ###

base_network_weights_path = network.get_weights_path()

try:
	print("Loading weights from {}".format(base_network_weights_path))
	model_rpn.load_weights(base_network_weights_path, by_name=True)
	print("Loaded weights!")
except:
	print("Could not load pretrained model weights.")


### Model Compilation ###

optimizer = Adam(lr = 1e-5, clipnorm = 1e-3)
model_rpn.compile(
                  optimizer = optimizer, 
                  loss = [losses.rpn_loss_cls(n_anchors), losses.rpn_loss_regr(n_anchors)]
                )
model_rpn.summary()


import time

### Training Parameters ###

n_epoch_length = args.n_epoch_length
n_epochs = args.n_epochs
iter_num = 0

### Monitoring Training Losses and Accuracy ###

# losses = np.zeros((n_epoch_length, 5))
# rpn_accuracy_rpn_monitor = []
# rpn_accuracy_for_epoch = []
# best_loss = np.Inf
# class_mapping_inv = {v: k for k, v in class_mapping.items()}


### Callbacks ###

model_save_path = args.model_save_path
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

model_save_path = os.path.join(model_save_path, "rpn") + "-" + base_network 
checkpoint_callback = ModelCheckpoint(
                    filepath = model_save_path + "-epoch-{epoch:02d}-loss-{loss:.2f}.hdf5", 
                    monitor = "loss", 
                    verbose = 1, 
                    save_best_only = True, 
                    save_weights_only = True,
                    mode = "auto", 
                    period = 1
                )

callbacks = [checkpoint_callback]

### Training ###

print("Training has started....")
start_time = time.time()
history = model_rpn.fit_generator(
                        generator = train_data_generator,
                        epochs = n_epochs, 
                        steps_per_epoch = n_epoch_length, 
                        callbacks = callbacks
                        )
end_time = time.time()
print("Training took {} seconds.".format(end_time - start_time))

### Logging Training History ###

loss_history = history.history["loss"]
numpy_loss_history = np.array(loss_history)
np.savetxt(model_save_path + "-loss_history.txt", numpy_loss_history, delimiter = ",")