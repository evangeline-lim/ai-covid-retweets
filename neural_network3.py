import math
import os
import joblib

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers.core import Dropout

# Seed value
# Apparently you may use different seed values at each stage
seed_value= 0

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
# import tensorflow as tf
# tf.random.set_seed(seed_value)
# for later versions: 
tf.compat.v1.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
# from keras import backend as K
# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)
# for later versions:

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

print("Num GPUs Available", len(tf.config.experimental.list_physical_devices('GPU')))

# Read dataset
dataset_1 = pd.read_csv('data/100_retweets.csv')
x1_1 = np.column_stack((np.array(dataset_1['followers']),
                      np.array(dataset_1['friends']),
                      np.array(dataset_1['favorites']),
                      np.array(dataset_1['entities']),
                      np.array(dataset_1['mentions']),
                      np.array(dataset_1['hashtags']),
                      np.array(dataset_1['urls']),
                      np.array(dataset_1['sentistrength'])))
y1_1= np.array(dataset_1['retweets'])
x_train_1, x_val_1, y_train_1, y_val_1 = train_test_split(x1_1, y1_1)

# Set hidden layers for the 4 different groups
training_data_samples_1 = len(x_train_1)

# Same for all different retweet groups
factor_ = 10
input_neurons1 = x1_1.shape[1] 

output_neurons = 1

# Different number of hidden layers for different retweet groups
hidden_layer_1 = int(training_data_samples_1/(factor_*(input_neurons1+output_neurons)))

#Define sequential model
model1 = Sequential()
model1.add(Dense(input_neurons1, input_dim=input_neurons1, kernel_initializer='normal', activation='relu'))
model1.add(Dense(hidden_layer_1, activation='relu'))
model1.add(Dense(1, activation='linear'))
model1.summary()

# Set Scalers
y_train_1 = np.reshape(y_train_1, (-1,1))
y_val_1 = np.reshape(y_val_1, (-1,1))

scaler_x_3 = MinMaxScaler()
scaler_y_3 = MinMaxScaler()

xtrain_scale_1 = scaler_x_3.fit_transform(x_train_1)
xval_scale_1 = scaler_x_3.transform(x_val_1)

ytrain_scale_1 = scaler_y_3.fit_transform(y_train_1)
yval_scale_1 = scaler_y_3.transform(y_val_1)

joblib.dump(scaler_x_3, 'data/scaler_x_3.save') 
joblib.dump(scaler_y_3, 'data/scaler_y_3.save')
print('Saved scalers to "data/"')

# Train model
model1.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
history_1 = model1.fit(xtrain_scale_1, ytrain_scale_1, epochs=20, batch_size=150, verbose=1, validation_split=0.2)

# Plot losses
print(history_1.history.keys())

plt.plot(history_1.history['loss'])
plt.plot(history_1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

predictions_1 = model1.predict(xval_scale_1)
predictions_1 = scaler_y_3.inverse_transform(predictions_1)

y_val_1_ = y_val_1[np.logical_not(np.isnan(y_val_1))]
predictions__1 = predictions_1[np.logical_not(np.isnan(predictions_1))]

print('MAE', mean_absolute_error(y_val_1_, predictions__1))
print('MSE', math.sqrt(mean_squared_error(y_val_1_, predictions__1)))
print('Mean Y value', np.mean(y_val_1_))
print('Mean Y predictions', np.mean(predictions__1))

# serialize model to JSON
model_json = model1.to_json()
with open("data/model3.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model1.save_weights("data/model3.h5")
print("Saved model to disk")