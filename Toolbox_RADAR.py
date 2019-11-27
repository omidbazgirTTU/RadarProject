# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:09:01 2019

@author: obazgir
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy import r_
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from time import time
from sklearn.model_selection import train_test_split


# Format image data with one-hot-encoding...
def FormatImageData(DSobjDF):
    te = time()
    
    N_files, N_class = len(DSobjDF), len(DSobjDF['class'].unique())   
    
    Im_data = [ ]
    Im_label = np.zeros((N_files, N_class))                                     # One-hot-encoding label matrix
    for i in tqdm(range(N_files)):
        filepath, filelabel, fileclass = DSobjDF.iloc[i, :]
        Im  = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)                        # Read image data
        
        # Check image size consistency...
        if i == 0:
            Im_res = Im.shape
        elif Im.shape != Im_res:
            print('All images must have the same size!!!')
        
        # Save data...
        Im_data.append(np.array(Im))
        Im_label[i, fileclass] = 1                                              # Assign label
    
    Im_data = np.reshape(Im_data, newshape = [-1, Im_res[0], Im_res[1], 1])
    Im_data = Im_data.astype(float) / 255.0
    DATAobj = [Im_data, Im_label]
    
    te = time() - te;    print('Elapsed time = %0.4f sec.\n' % te)
    
    return DATAobj


def ExtractData(TrainDSobj, TestDSobj, split_size = 0.2):   
    te = time()
    
    # Training - validation split...
    N_train = len(TrainDSobj);      print('\nTotal training data size = %d' % N_train)
    split   = train_test_split(TrainDSobj, TrainDSobj['class'], test_size = split_size, 
                               stratify = TrainDSobj['class'], random_state = None)
    Imds_train, Imds_valid, Imds_test = split[0], split[1], TestDSobj
    n_train, n_valid, n_test = Imds_train.shape[0], Imds_valid.shape[0], Imds_test.shape[0]
    
    # Check datasets...
    print('Train - Validation split = %d%% - %d%% (%d - %d)' 
          % (np.round(n_train / N_train * 100), np.round(n_valid / N_train * 100), n_train, n_valid))
    print('Training data 0-1 fraction = [%0.4f, %0.4f]'   
          % ((Imds_train['class'] == 0).mean(), (Imds_train['class'] == 1).mean()))
    print('Validation data 0-1 fraction = [%0.4f, %0.4f]' 
          % ((Imds_valid['class'] == 0).mean(), (Imds_valid['class'] == 1).mean()))
    
    
    # Read data...
    X_train, y_train = FormatImageData(Imds_train)
    X_valid, y_valid = FormatImageData(Imds_valid)
    X_test,  y_test  = FormatImageData(Imds_test)
    
    print('\nTraining data shape =', X_train.shape)
    print('Validation data shape =', X_valid.shape)
    print('Test data shape =',       X_test.shape)
    
    DSSobj = {'Training': (X_train, y_train), 'Validation': (X_valid, y_valid), 'Test': (X_test, y_test)}   # Return object
    
    te = time() - te;    print('\nTotal time elapsed = %0.4f sec.\n' % te)
    
    return DSSobj