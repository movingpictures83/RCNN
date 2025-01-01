#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from pandas import concat
from pandas import read_csv
from helper import series_to_supervised, stage_series_to_supervised


# In[3]:

import PyPluMA
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# In[11]:

class RCNNPlugin:
 def input(self, inputfile):
  self.dataset = pd.read_csv(inputfile, index_col=0)
 def run(self):
     pass
 def output(self, outputfile):
  self.dataset.fillna(0, inplace=True)
  data = self.dataset
  n_hours = 24*7
  K = 24
  stages = self.dataset[['WS_S1', 'TWS_S25A', 'TWS_S25B', 'TWS_S26']]
  stages_supervised = series_to_supervised(stages, n_hours, K)
  non_stages = data[['WS_S4', 'FLOW_S25A', 'FLOW_S25B', 'FLOW_S26', 'PUMP_S26', 'PUMP_S25B', 'MEAN_RAIN']]
  non_stages_supervised = series_to_supervised(non_stages, n_hours-1, 1)
  non_stages_supervised_cut = non_stages_supervised.iloc[K:, :]
  n_features = stages.shape[1] + non_stages.shape[1]   # 1 rainfall + FGate_S25A + FGate_S25B + FGate_S26 + 8WS + PUMP_S26
  non_stages_supervised_cut.reset_index(drop=True, inplace=True)
  stages_supervised.reset_index(drop=True, inplace=True)

  all_data = concat([
                   non_stages_supervised_cut.iloc[:, :],
                   stages_supervised.iloc[:, :]],
                   axis=1)
  all_data = all_data.values
  n_train_hours = int(len(all_data)*0.8)
  train = all_data[:n_train_hours, :]
  test = all_data[n_train_hours:, :]
  n_obs = n_hours * n_features
  train_X, train_y = train[:, :n_obs], train[:, -stages.shape[1]*K:]
  test_X, test_y = test[:, :n_obs], test[:, -stages.shape[1]*K:]
  scaler = MinMaxScaler(feature_range=(0, 1))
  train_X = scaler.fit_transform(train_X)
  train_y = scaler.fit_transform(train_y)
  test_X = scaler.fit_transform(test_X)
  test_y = scaler.fit_transform(test_y)
  train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
  test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
  n_outputs = test_y.shape[1]
  model_rcnn_60 = keras.Sequential()
  model_rcnn_60.add(layers.SimpleRNN(128, activation="relu", return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
  model_rcnn_60.add(layers.Conv1D(filters=64, kernel_size=2, activation="relu"))
  model_rcnn_60.add(layers.MaxPooling1D(pool_size=2))
  model_rcnn_60.add(layers.Flatten())
  model_rcnn_60.add(layers.Dense(test_y.shape[1]))
  model_rcnn_60.summary()
  lr = 0.00001
  EPOCHS = 5
  model_rcnn_60.compile(
              optimizer=Adam(learning_rate=lr, decay=lr/EPOCHS), 
              loss='mse',
              metrics=['mae'])

  es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=500)
  mc = ModelCheckpoint(PyPluMA.prefix()+"/saved_model/rcnn_24h.h5", monitor='val_mae', mode='min', verbose=2, save_best_only=True)


  history = model_rcnn_60.fit(train_X, train_y,
                            batch_size=512,
                            epochs=EPOCHS,
                            validation_data=(test_X, test_y),
                            verbose=2,
                            shuffle=False,
                           callbacks=[es, mc])

  plt.rcParams["figure.figsize"] = (8, 6)
  plt.plot(history.history['loss'], label='train')
  plt.plot(history.history['val_loss'], label='test')
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.xlabel('Epoch', fontsize=16)
  plt.ylabel('Loss', fontsize=16)
  plt.legend(fontsize=14)
  plt.title("Training loss vs Testing loss", fontsize=18)
  # plt.savefig('graph/rnn_loss.png', dpi=300)
  plt.show()

  from tensorflow.keras.models import load_model

  model_load = load_model(PyPluMA.prefix()+"/saved_model/rcnn.h5")

  import time

  yhat = model_load.predict(test_X)

  inv_yhat = scaler.inverse_transform(yhat)
  inv_y = scaler.inverse_transform(test_y)

  inv_yhat = pd.DataFrame(inv_yhat)
  inv_y = pd.DataFrame(inv_y)

  error_abs = abs(inv_yhat - inv_y)
  error = inv_yhat - inv_y
  error_19_20 = error.iloc[-17544:, :]

  print('MAE = {}'.format(float("{:.4f}".format(mae(inv_yhat.iloc[:, :], inv_y.iloc[:, :])))))
  print('RMSE = {}'.format(float("{:.4f}".format(sqrt(mse(inv_yhat.iloc[:, :], inv_y.iloc[:, :]))))))

  S1_index = [i for i in range(0, K*4, 4)]
  S25A_index = [i+1 for i in range(0, K*4, 4)]
  S25B_index = [i+2 for i in range(0, K*4, 4)]
  S26_index = [i+3 for i in range(0, K*4, 4)]

  locations = ['WS_S1', 'TWS_S25A', 'TWS_S25B', 'TWS_S26']
  loc_index = [S1_index, S25A_index, S25B_index, S26_index]
  for i in range(len(locations)):
    print('Errors of {}'.format(locations[i]))
    print('MAE = {}'.format(float("{:.4f}".format(mae(inv_yhat.iloc[:, loc_index[i]], inv_y.iloc[:, loc_index[i]])))))
    print('RMSE = {}'.format(float("{:.4f}".format(sqrt(mse(inv_yhat.iloc[:, loc_index[i]], inv_y.iloc[:, loc_index[i]]))))))
    print('====================')

  from tensorflow.keras.models import load_model
  models = ['mlp', 'rnn', 'lstm', 'cnn', 'rcnn_24h']

  for i in range(len(models)):
    print("========= {} =========".format(models[i]))
    saved_model = load_model(PyPluMA.prefix()+"/saved_model/{}.h5".format(models[i]))
    yhat = saved_model.predict(test_X)
    inv_yhat = scaler.inverse_transform(yhat)
    inv_y = scaler.inverse_transform(test_y)
    inv_yhat = pd.DataFrame(inv_yhat)
    inv_y = pd.DataFrame(inv_y)
    error = inv_y - inv_yhat
    print('1h MAE = {}'.format(float("{:.6f}".format(mae(inv_yhat.iloc[:, 0:4], inv_y.iloc[:, 0:4])))))
    print('1h RMSE = {}'.format(float("{:.6f}".format(sqrt(mse(inv_yhat.iloc[:, 0:4], inv_y.iloc[:, 0:4]))))))
    print('8h MAE = {}'.format(float("{:.6f}".format(mae(inv_yhat.iloc[:, 28:32], inv_y.iloc[:, 28:32])))))
    print('8h RMSE = {}'.format(float("{:.6f}".format(sqrt(mse(inv_yhat.iloc[:, 28:32], inv_y.iloc[:, 28:32]))))))
    print('16h MAE = {}'.format(float("{:.6f}".format(mae(inv_yhat.iloc[:, 60:64], inv_y.iloc[:, 60:64])))))
    print('16h RMSE = {}'.format(float("{:.6f}".format(sqrt(mse(inv_yhat.iloc[:, 60:64], inv_y.iloc[:, 60:64]))))))
    print('24h MAE = {}'.format(float("{:.6f}".format(mae(inv_yhat.iloc[:, 92:96], inv_y.iloc[:, 92:96])))))
    print('24h RMSE = {}'.format(float("{:.6f}".format(sqrt(mse(inv_yhat.iloc[:, 92:96], inv_y.iloc[:, 92:96]))))))
    print('All MAE = {}'.format(float("{:.6f}".format(mae(inv_yhat.iloc[:, :], inv_y.iloc[:, :])))))
    print('All RMSE = {}'.format(float("{:.6f}".format(sqrt(mse(inv_yhat.iloc[:, :], inv_y.iloc[:, :]))))))





