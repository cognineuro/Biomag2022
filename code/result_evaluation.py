#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 15:09:42 2022

@author: xuyierjing
"""
import numpy as np
from numpy import load, save
import matplotlib.pyplot as plt

# labels
# yoko (A): 0-control; 1-dementia; 2-mci
# richo (B): 3-control; 4-dementia; 5-mci
# In[] xgboost-richo
result_richo_xgboost = load('xgboost_sep4_test_richo_result_balanced.npy',allow_pickle=True)

no_segment = 4*4 # 4 hyperpara x 4 segment
record_richo_xgb = list()
for sub in range(20):
    count_control = (result_richo_xgboost[:,sub*4:sub*4+4] == 3).sum()
    count_dementia = (result_richo_xgboost[:,sub*4:sub*4+4] == 4).sum()
    count_mci = (result_richo_xgboost[:,sub*4:sub*4+4] == 5).sum()
    record_richo_xgb.append([count_control/no_segment, count_dementia/no_segment, count_mci/no_segment]) 

np.save('result_xgb_richo', record_richo_xgb)
for index, value in enumerate(record_richo_xgb):
    print(index,value.index(max(value)))#print(index, max(value))
    
# In[] xgboost-yoko
result_yoko_xgboost = load('xgboost_sep4_test_yoko_result_balanced.npy',allow_pickle=True)

no_segment = 3*4 # 4 hyperpara x 4 segment
record_yoko_xgb = list()
for sub in range(22):
    count_control = (result_yoko_xgboost[:,sub*4:sub*4+4] == 0).sum()
    count_dementia = (result_yoko_xgboost[:,sub*4:sub*4+4] == 1).sum()
    count_mci = (result_yoko_xgboost[:,sub*4:sub*4+4] == 2).sum()
    record_yoko_xgb.append([count_control/no_segment, count_dementia/no_segment, count_mci/no_segment]) 

np.save('result_xgb_yoko', record_yoko_xgb)

for index, value in enumerate(record_yoko_xgb):
    print(index,value.index(max(value)))#print(index, max(value))

# In[] Siamese-yoko (MCI)
result_yoko_siam = load('Siamese_sep4_test_yoko_result.npy',allow_pickle=True).T
label_yoko_siam = load('Siamese_sep4_test_yoko_labels.npy',allow_pickle=True)

########### analysis I
# no_pair = 100*4 # 100 rep x 4 pairs
# record_yoko_siam_pair = list()
# for pair in range(88):
#     count_dist_pair = result_yoko_siam[np.squeeze((np.where(label_yoko_siam[:,2]== pair))),:].sum()
#     record_yoko_siam_pair.append(count_dist_pair/no_pair) 

# no_segment = 4 #  4 segment
# record_yoko_siam = list()
# for sub in range(22):
#     count_dist = sum(record_yoko_siam_pair[sub*4:sub*4+4])
#     record_yoko_siam.append(count_dist/no_segment) 

# # baseline distance
# count_dist_base = result_yoko_siam[np.squeeze((np.where(label_yoko_siam[:,0]== 1))),:]
# record_yoko_siam_base = np.mean(count_dist_base, axis = 1)

# # select the potential MCI subs in the yoko site
# record_yoko_mci =[list(record_yoko_siam_pair[i]) for i in range(len(record_yoko_siam_pair)) if i <= max(record_yoko_siam_base)]

########## analysis II
# count_mci_all = []
# for pair in range(88):
#     count_mci = []
#     for rep in range(100):
#         count_dist_pair = result_yoko_siam[np.squeeze((np.where(label_yoko_siam[:,2]== pair))),rep]
#         count_base = result_yoko_siam[np.squeeze((np.where(label_yoko_siam[:,0]== 1))),rep]
#         count_mci += [np.squeeze(np.where(count_dist_pair <= max(count_base))).size]
#     count_mci_all += [np.sum(count_mci)]
#     quotients = [number / 400 for number in count_mci_all]

count_mci_all = []
for pair in range(88):
    count_mci = []
    for rep in range(100):
        count_dist_pair = result_yoko_siam[np.squeeze((np.where(label_yoko_siam[:,2]== pair))),rep]
        # baseline distance
        count_base = result_yoko_siam[np.squeeze((np.where(label_yoko_siam[:,0]== 1))),rep]
        # compare the testng samples with the baseline (95% CI)
        count_mci += [np.squeeze(np.where(count_dist_pair <= max(count_base+np.std(count_base)*1.96))).size]
    count_mci_all += [np.sum(count_mci)]
quotients = [number / 400 for number in count_mci_all]

quotients_sub = []
for sub in range(22):    
    quotients_sub += [np.mean(quotients[sub*4:sub*4+4])]
    
# plot the distance
mci_sub = [44,45,46,47]#[32,33,34,35]#[4,5,6,7]
proto_sub = [0,1,2,91,92,181]
bins = np.linspace(-0.65, -0.4, 400)
plt.hist(result_yoko_siam[proto_sub,:].ravel(), bins=bins, alpha=0.5, label="SiteA_mci prototype")
plt.hist(result_yoko_siam[np.squeeze(np.where(np.in1d(label_yoko_siam[:,2], mci_sub))),:].ravel(), bins=bins, alpha=0.5, label="Test data")
plt.xlabel("Data distance", size=14)
plt.ylabel("Count", size=14)
#plt.title("Data distance")
plt.legend(loc='upper right')
plt.show()
plt.savefig("Siamese_sep4_test_yoko_v2_sub22.png")    

# In[] CNN for both yoko and richo
richo_indx = [3, 4, 6, 8, 11, 16, 17, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33, 34, 42]; 
richo_indx = [x - 1 for x in richo_indx]
yoko_indx = list(range(42))
for i in richo_indx:
    yoko_indx.remove(i)

result_CNN = load('CNN_sep4_test_result.npy',allow_pickle=True)
# result_richo_all = np.concatenate([result_richo_xgboost, result_richo_CNN], axis=1)

no_segment = 50*4 # 50 rep x 4 segment
record_CNN = list()
for sub in range(42):
    count_yoko_control = (result_CNN[:,sub*4:sub*4+4] == 0).sum()
    count_yoko_dementia = (result_CNN[:,sub*4:sub*4+4] == 1).sum()
    count_yoko_mci = (result_CNN[:,sub*4:sub*4+4] == 2).sum()
    count_richo_control = (result_CNN[:,sub*4:sub*4+4] == 3).sum()
    count_richo_dementia = (result_CNN[:,sub*4:sub*4+4] == 4).sum()
    count_richo_mci = (result_CNN[:,sub*4:sub*4+4] == 5).sum()
    record_CNN.append([count_yoko_control/no_segment, count_yoko_dementia/no_segment, count_yoko_mci/no_segment,
                       count_richo_control/no_segment, count_richo_dementia/no_segment, count_richo_mci/no_segment]) 

np.save('result_CNN', record_CNN)
record_richo_CNN =[list(record_CNN[i]) for i in range(len(record_CNN)) if i in richo_indx]
for index, value in enumerate(record_richo_CNN):
    print(index,value.index(max(value)))#print(index, max(value))

record_yoko_CNN =[list(record_CNN[i]) for i in range(len(record_CNN)) if i in yoko_indx]
for index, value in enumerate(record_yoko_CNN):
    print(index, max(value))#print(index,value.index(max(value)))
 

