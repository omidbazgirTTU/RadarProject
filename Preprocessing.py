# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 11:54:27 2019

@author: obazgir
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy import r_
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from time import time
from DirDataStore import DirDataStore




#%% PATH & FILENAMES...

#%cls
PATH    = os.getenv('HOMEPATH') + '\\Dropbox\\RadarImages\\Dhruba\\'
DATADIR = ['1p', '1p_10212018', '2p_mod', '3p', '3p_train']
FILTER  = ['walk', 'gym', 'gun', 'walkangle', 'gymangle', 'gunangle', 'gymgun', 'gymgungym']

# List dataset paths...
Imds_header = ['root', 'label']
Imds_1p  = pd.DataFrame(DirDataStore(PATH + '1p',          filter = True, filter_dict = FILTER), index = Imds_header).T
Imds_1p2 = pd.DataFrame(DirDataStore(PATH + '1p_10212018', filter = True, filter_dict = FILTER), index = Imds_header).T
Imds_2p  = pd.DataFrame(DirDataStore(PATH + '2p_mod',      filter = True, filter_dict = FILTER), index = Imds_header).T
Imds_3p  = pd.DataFrame(DirDataStore(PATH + '3p',          filter = True, filter_dict = FILTER), index = Imds_header).T
