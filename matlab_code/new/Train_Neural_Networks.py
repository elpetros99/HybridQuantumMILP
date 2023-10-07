import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import time
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import time
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import os
import tensorflow_model_optimization as tfmot

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

overall_start_time = time.time()

with open (".\path_python.txt", "r") as myfile:
    path=myfile.readlines()
path = path[0]
print(path)

raw_dataset_input = pd.read_csv(path + "NN_input.csv", delimiter=",",header=None)
raw_dataset_output = pd.read_csv(path + "NN_output.csv", delimiter=",",header=None)

input_len = raw_dataset_input.shape[1]
print(input_len)
output_len = raw_dataset_output.shape[1]
print(output_len)

nr_samples = raw_dataset_output.shape[0]
print(nr_samples)
raw_dataset=pd.concat([raw_dataset_input,raw_dataset_output],axis=1,ignore_index=1)

dataset = raw_dataset.copy()


train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)


train_labels = train_dataset.iloc[:,input_len:output_len+input_len]
test_labels = test_dataset.iloc[:,input_len:output_len+input_len]
train_features = train_dataset.iloc[:,0:input_len]
test_features = test_dataset.iloc[:,0:input_len]

data_features = dataset.iloc[:,0:input_len]
data_labels = dataset.iloc[:,input_len:output_len+input_len]

np.savetxt(path+'features_test.csv',test_features.to_numpy(),delimiter=',')
np.savetxt(path+'labels_test.csv',test_labels.to_numpy(),delimiter=',')
np.savetxt(path+'features_train.csv',train_features.to_numpy(),delimiter=',')
np.savetxt(path+'labels_train.csv',train_labels.to_numpy(),delimiter=',')


# 50 neurons and 3 hidden layers
nr_neurons = 50

#pruning_params = {
#    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.5, 0)}

# gradual pruning from 0% to 80% sparsity
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.00,
                                                   final_sparsity=0.80,
                                                   begin_step=500,
                                                   end_step=2500,
                                                   frequency=100)
}

input_nn = Input(shape=(len(train_features.keys()),))
layer1 = tfmot.sparsity.keras.prune_low_magnitude(layers.Dense(nr_neurons, activation='relu'),**pruning_params)(input_nn)
layer2 = tfmot.sparsity.keras.prune_low_magnitude(Dense(nr_neurons, activation='relu'),**pruning_params)(layer1)
layer3 = tfmot.sparsity.keras.prune_low_magnitude(Dense(nr_neurons, activation='relu'),**pruning_params)(layer2)
predictions = tfmot.sparsity.keras.prune_low_magnitude(Dense([len(train_labels.keys())][0]),**pruning_params)(layer3)
model = Model(inputs=input_nn, outputs=predictions)


EPOCHS = 250

# We split the training set into 200 batches
batch_size_dyn = int(np.maximum(int(nr_samples/200),5))

# early stopping if we observe no improvement for 50 epochs
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50,restore_best_weights=True,min_delta=0.0)

optimizer = tf.keras.optimizers.Adam(0.001)

callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            es
        ]
        
    
start_time = time.time()
                    
# STANDARD TRAINING
model.compile(loss='mse',optimizer=optimizer, metrics=['mse']) 

end_time = time.time()
print("Time usage compiling: " + str(timedelta(seconds=int(round(end_time - start_time)))))
start_time = time.time()
            
history = model.fit(train_features.to_numpy(), train_labels.to_numpy(),
epochs=EPOCHS,verbose=0,callbacks=callbacks,validation_data=(test_features.to_numpy(),test_labels.to_numpy()),batch_size=batch_size_dyn)

end_time = time.time()
print("Time usage training: " + str(timedelta(seconds=int(round(end_time - start_time)))))
start_time = time.time()
        
loss_test= model.evaluate(test_features.to_numpy(), test_labels.to_numpy(),verbose=0)
loss_train= model.evaluate(train_features.to_numpy(), train_labels.to_numpy(),verbose=0)
loss_overall = model.evaluate(data_features.to_numpy(), data_labels.to_numpy(),verbose=0)
print("LOSSES")            
print(loss_test)
print(loss_train)
print(loss_overall)


# save model weights --> this allows us later to restore the NN without re-training
model.save_weights(path+'my_model_weights.h5')

# save weights for post-process (compute worst-case guarantees) with Matlab
adv_model=tfmot.sparsity.keras.strip_pruning(model)
for j in range(0, 4):
            weights = adv_model.get_weights()[2*j]
            biases = adv_model.get_weights()[2*j+1]
            np.savetxt(path+'W'+str(j)+'.csv',weights, fmt='%s', delimiter=',')
            np.savetxt(path+'b'+str(j)+'.csv',biases, fmt='%s', delimiter=',')
                


