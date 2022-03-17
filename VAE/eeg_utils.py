# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 18:38:17 2022

@author: Kellen Cheng
"""

# Import Statements
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
tf.compat.v1.enable_eager_execution()
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

# %% Train Model Functions
def find_optimal_model(model1, X_train, y_train, X_valid, y_valid,
                       X_test, y_test, filename, iterations, 
                       prev_best=-1.0, batches=64):
    count = 0
    for i in range(iterations): # Set maximum to 200 epochs
        print("EPOCH:", str(i+1))
        
        # Calculate train, validation, and test accuracies per epoch
        results = model1.fit(X_train, y_train, batch_size=batches, 
                             epochs=1, validation_data=(X_valid, y_valid), 
                             verbose=True)
        valid_score = model1.evaluate(X_test, y_test)
        
        # Replace model with newer version if better
        if (valid_score[1] > prev_best):
            model1.save(filename)
            count = i + 1
            prev_best = valid_score[1]
        pass
        
    return prev_best, count

# %% Preprocess Data Functions
def preprocess_data(X_train_valid1, y_train_valid1, 
                    X_test1, y_test1, data_config):
    
    '''
    data_config = {"Duplicate": int, "Mu": float, "Sigma": float,
    "Sample": bool, "Sample_Step": int, "Attention": bool, "Swap": bool}
    '''
    # Trim class labels to the range [0, 1, 2, 3]
    y_train_valid1 = y_train_valid1 - 769
    y_test1 = y_test1 - 769
    
    # Duplicate our data with Gaussian noise, if specified
    mu = data_config["Mu"]
    sigma = data_config["Sigma"]
    temp_x = [X_train_valid1]
    temp_y = [y_train_valid1]
    
    for i in range(data_config["Duplicate"] - 1):
        vals = X_train_valid1 + (np.random.normal(mu, sigma, X_train_valid1.shape))
        temp_x.append(vals)
        temp_y.append(y_train_valid1)
        pass
    
    # Aggregate all data and cast to Numpy array
    X_train_valid1 = layers.Concatenate(axis=0)(temp_x)
    y_train_valid1 = layers.Concatenate(axis=0)(temp_y)
    X_train_valid1 = X_train_valid1.numpy()
    y_train_valid1 = y_train_valid1.numpy()
    
    # Reshape our raw data for neural network processing
    X_train_valid1 = X_train_valid1.reshape((X_train_valid1.shape)[0],
                                                (X_train_valid1.shape)[1],
                                                (X_train_valid1.shape)[2],
                                                1)
    X_test1 = X_test1.reshape((X_test1.shape)[0],
                              (X_test1.shape)[1],
                              (X_test1.shape)[2],
                              1)
    
    # Swap the axes in our raw data
    if (data_config["Swap"] == True):
        X_train_valid1 = np.swapaxes(np.swapaxes(X_train_valid1, 1, 2), 2, 3)
        X_test1 = np.swapaxes(np.swapaxes(X_test1, 1, 2), 2, 3)
    
    # Convert ground truths to categorical values
    y_train_valid1 = to_categorical(y_train_valid1, 4)
    y_test1 = to_categorical(y_test1, 4)
    
    
    # Accentuate highest variation areas, if required
    if (data_config["Attention"] == True):
        variations = get_variation(X_train_valid1, y_train_valid1)
        
        for i in range(1000):
            X_train_valid1[:, i, 0, :] *= variations[i]**2
            X_test1[:, i, 0, :] *= variations[i]**2
            pass

    # Retrieve our preprocessed data
    X_train_valid = X_train_valid1
    y_train_valid = y_train_valid1
    X_test = X_test1
    y_test = y_test1
    
    return X_train_valid, y_train_valid, X_test, y_test

def vae_preprocess_data(X_train_valid1, y_train_valid1, X_test1, y_test1, data_config):
    '''
    data_config = {"Duplicate": int, "Mu": float, "Sigma": float}
    '''
    # Trim class labels to the range [0, 1, 2, 3]
    y_train_valid1 = y_train_valid1 - 769
    y_test1 = y_test1 - 769
    
    # Duplicate our data with Gaussian noise, if specified
    mu = data_config["Mu"]
    sigma = data_config["Sigma"]
    temp_x = [X_train_valid1]
    temp_y = [y_train_valid1]
    
    for i in range(data_config["Duplicate"] - 1):
        vals = X_train_valid1 + (np.random.normal(mu, sigma, X_train_valid1.shape))
        temp_x.append(vals)
        temp_y.append(y_train_valid1)
        pass
    
    # Aggregate all data and cast to Numpy array
    X_train_valid1 = layers.Concatenate(axis=0)(temp_x)
    y_train_valid1 = layers.Concatenate(axis=0)(temp_y)
    X_train_valid1 = X_train_valid1.numpy()
    y_train_valid1 = y_train_valid1.numpy()
    
    # Retrieve our preprocessed data
    X_train_valid = X_train_valid1
    y_train_valid = y_train_valid1
    X_test = X_test1
    y_test = y_test1
    
    # Trim our data based on the time point samples
    X_train_valid = X_train_valid[:, :, :data_config["Time_Point"]]
    X_test = X_test[:, :, :data_config["Time_Point"]]
    
    # Convert truths to categorical truths for loss function
    y_train_valid = to_categorical(y_train_valid)
    y_test = to_categorical(y_test)
    
    return X_train_valid, y_train_valid, X_test, y_test

def sampling(params):
    z_mu, z_var = params
    batch = tf.shape(z_mu)[0]
    dim = tf.shape(z_var)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mu + epsilon * tf.exp(0.5 * z_var)

def get_variation(X_train_valid1, y_train_valid1):
    reach_left = []
    reach_right = []
    move_feet = []
    move_tongue = []
    
    for i in range((X_train_valid1.shape)[0]):
        data = X_train_valid1[i, :, :, :]
        idx = np.where(y_train_valid1[i] == 1)
        
        if (idx[0][0] == 3):
            move_tongue.append(data)
        elif (idx[0][0] == 2):
            move_feet.append(data)
        elif (idx[0][0] == 1):
            reach_right.append(data)
        else:
            reach_left.append(data)
            
    avg_reach_left = np.squeeze(np.mean(np.array(reach_left), axis=0))
    avg_reach_right = np.squeeze(np.mean(np.array(reach_right), axis=0))
    avg_move_feet = np.squeeze(np.mean(np.array(move_feet), axis=0))
    avg_move_tongue = np.squeeze(np.mean(np.array(move_tongue), axis=0))
    
    mean_avg_reach_left = np.mean(avg_reach_left, axis=1)
    mean_avg_reach_right = np.mean(avg_reach_right, axis=1)
    mean_avg_move_feet = np.mean(avg_move_feet, axis=1)
    mean_avg_move_tongue = np.mean(avg_move_tongue, axis=1)
    
    means = np.array([mean_avg_reach_left, mean_avg_reach_right,
                      mean_avg_move_feet, mean_avg_move_tongue])
    variations = np.var(means, axis=0)
    variations = variations.reshape((variations.shape)[0], 1, 1)
    return variations

# %% Test Cell
pass