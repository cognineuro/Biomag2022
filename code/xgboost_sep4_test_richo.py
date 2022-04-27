# In[] clear all variables
#from IPython import get_ipython
#get_ipython().magic('reset -sf')

# In[] Preprocessing
from xgboost import XGBClassifier

from sklearn.utils import class_weight

from numpy import load, save
import scipy.io as sio
import numpy as np

# In[] Load the training Dataset
power = sio.loadmat("/raid/xuyierjing/biomag_competition/training/data_SMH/meanpower_sep4_SMH.mat") # sep_data_final.csv
Hil = power['wr_Hilbert_relative_sep4']
Fft = power['wr_FFT_relative_sep4']
Hil3 = Hil[:,160:640]

temp = sio.loadmat('/raid/xuyierjing/biomag_competition/training/biomarker/entropy/LZC/LZC_all_sep4.mat')
Lzc = temp['LZC_all_sep4']

labelname = '/raid/xuyierjing/biomag_competition/training/data_SMH/labels_meanpower_sep4.npy'
y_train = load(labelname) 
y_train = y_train.ravel() 

# data = np.concatenate((Lzc, Hil), axis = 1)
x_train = np.concatenate((Lzc, Hil3), axis = 1)

# Richo 
richo_control_orig = [2, 8, 9, 10, 14, 17, 18, 19, 20, 23, 31, 32, 35, 40, 49, 57, 77, 78, 79, 88, 90, 94, 95, 96, 97, 99, 100];
richo_dementia_orig = [2, 4, 6, 7, 8, 9, 11, 20, 22, 25];
richo_mci_orig     = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15];

richo_control1 = [4*(x-1)+1 for x in richo_control_orig]
richo_control2 = [x+1 for x in richo_control1]
richo_control3 = [x+2 for x in richo_control1]
richo_control4 = [x+3 for x in richo_control1]
richo_control = richo_control1 + richo_control2  + richo_control3  + richo_control4
richo_control.sort()

richo_dementia1 = [4*(x-1)+1 for x in richo_dementia_orig]
richo_dementia2 = [x+1 for x in richo_dementia1]
richo_dementia3 = [x+2 for x in richo_dementia1]
richo_dementia4 = [x+3 for x in richo_dementia1]
richo_dementia = richo_dementia1 + richo_dementia2  + richo_dementia3  + richo_dementia4
richo_dementia.sort()

richo_mci1 = [4*(x-1)+1 for x in richo_mci_orig]
richo_mci2 = [x+1 for x in richo_mci1]
richo_mci3 = [x+2 for x in richo_mci1]
richo_mci4 = [x+3 for x in richo_mci1]
richo_mci = richo_mci1 + richo_mci2  + richo_mci3  + richo_mci4
richo_mci.sort()

richo_dementia = [x+400 for x in richo_dementia]
richo_mci = [x+516 for x in richo_mci]
siteB = richo_control + richo_dementia + richo_mci
siteB  = [x - 1 for x in siteB]

x_train_richo = x_train[siteB,:]
y_train_richo = y_train[siteB]

# siteA = list(range(576))
# for i in siteB:
#     siteA.remove(i)

# training_data_yoko = training_data[siteA,:]
# training_labels_yoko = training_labels[siteA]

# In[] Load the testing Dataset
power = sio.loadmat('/raid/xuyierjing/biomag_competition/testing/testing_data/meanpower_sep4_SMH.mat') # sep_data_final.csv
Hil = power['wr_Hilbert_relative_sep4']
Fft = power['wr_FFT_relative_sep4']
Hil3 = Hil[:,160:640]

temp = sio.loadmat('/raid/xuyierjing/biomag_competition/testing/testing_data/LZC_sep4.mat')
Lzc = temp['LZC_sep4']

labelname = '/raid/xuyierjing/biomag_competition/testing/testing_data/labels_sep4.npy'
label_test = load(labelname) 
label_test = label_test.ravel() 

x_test = np.concatenate((Lzc, Hil3), axis = 1)
x_test_richo = x_test[np.squeeze(np.where(label_test ==1)),:]

# In[] load the best paramters and model traiing
classes_weights = class_weight.compute_sample_weight(
    class_weight='balanced',
    y=y_train_richo)
        
finetune_result = load('/raid/xuyierjing/biomag_competition/training/script_SMH/xgboost/XGB_finetune_sep4_LZCHil3_richo.npy',allow_pickle=True)
finetune_result[:,1].sort()
richo_param = finetune_result[9,5]
richo_param2 = finetune_result[3,5]
richo_param3 = finetune_result[8,5]

model = XGBClassifier(**richo_param, random_state = 20)
model.fit(x_train_richo, y_train_richo, sample_weight=classes_weights) #    
y_test_pred = model.predict(x_test_richo)

model2 = XGBClassifier(**richo_param2, random_state = 20)
model2.fit(x_train_richo, y_train_richo, sample_weight=classes_weights)
y_test_pred2 = model2.predict(x_test_richo)

model3 = XGBClassifier(**richo_param3, random_state = 20)
model3.fit(x_train_richo, y_train_richo, sample_weight=classes_weights)
y_test_pred3 = model3.predict(x_test_richo)

finetune_result = load('/raid/xuyierjing/biomag_competition/training/script_SMH/xgboost/XGB_finetune_sep4_LZCHil3_richo_CV.npy',allow_pickle=True)
finetune_result[:,0].sort()
richo_param4 = finetune_result[1,2]
model4 = XGBClassifier(**richo_param4, random_state = 20)
model4.fit(x_train_richo, y_train_richo, sample_weight=classes_weights)
y_test_pred4 = model3.predict(x_test_richo)

results = list([y_test_pred, y_test_pred2, y_test_pred3, y_test_pred4])
np.save('xgboost_sep4_test_richo_result_balanced', results)
