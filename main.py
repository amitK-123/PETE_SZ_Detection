import numpy as np
from mhrc_util import mhrc_training
from ibib_util import ibib_training
from utilities import start_training
import tensorflow as tf
import keras

# dataset_idx = int(input("Enter 0 to train model over IBIB Dataset and 1 to train over MHRC Dataset: "))
# sec = int(input("Enter Time Window: "))

dataset_idx = 0

sec_window = [1]

for sec in sec_window:
    
    print(f' Starting with {sec} seconds window')

    x_train,x_test,y_train,y_test = mhrc_training(sec) if dataset_idx == 1 else ibib_training(sec)

    model = start_training(x_train,x_test,y_train,y_test,dataset_idx,sec)
    print(f' Ended with {sec} seconds window')

# val = int(input("Enter 1 to save the model"))

# if val==1:
#     name = input('Enter Name of the Model')
#     model.save(f'./Saved Model/{name}.h5')
#     loaded_model = keras.models.load_model(f'./Saved Model/{name}.h5')
#     print(loaded_model)
