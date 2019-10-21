# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 14:40:21 2019

@author: Gonçalo
"""
import compute_mean_max
from tqdm import tqdm
import os
import json
import collections
import pandas as pd
import pickle
import numpy as np


DATA_ROOT='D:\CastelBranco\PAPER'

MAX_OM_CLIPS=21000 #maximum of instances per class for the openmic train set
MAX_PM2_CLIPS=40000 #maximum of instances per class for the final dataset so the 
                    #number of clips per class is balanced
SIL_THRESHOLD=0.5
AS_INST_THRESHOLD=0.5
OM_INST_THRESHOLD=0.5
        
if __name__ == '__main__':
    
    with open(os.path.join(DATA_ROOT, 'class-map.json'), 'r') as f:
        class_map = json.load(f)
    
    split_test = pd.read_csv(os.path.join(DATA_ROOT,'test_split.csv'), 
                          header=None, squeeze=True)

    pmic_test_set = set(split_test)
        
    OM_all_X,OM_all_Y,OM_all_N,N=compute_mean_max.get_Omic_data()
    train_idx,test_idx=compute_mean_max.get_omic_splits(OM_all_N)
    OM_X_ALL_TRAIN=OM_all_X[train_idx]
    OM_Y_ALL_TRAIN=OM_all_Y[train_idx]
    
    om_scores=compute_mean_max.get_scores_NN('NN_ALL',OM_X_ALL_TRAIN)
    PM2_X=[]
    PM2_Y=[]
    for inst_idx,inst in tqdm(enumerate(class_map)):
        num_ins=0
        for idx, score in enumerate(om_scores):
            if score[inst_idx]>0.5 and OM_Y_ALL_TRAIN[idx][inst_idx]>0.5 and num_ins < MAX_OM_CLIPS:
                PM2_X.append(OM_X_ALL_TRAIN[idx])
                PM2_Y.append(str(inst))
                num_ins=num_ins+1

    counts=collections.Counter(PM2_Y)
    print('Omic counts:')
    print(counts)
    num_sil=0
    for inst_idx,inst in tqdm(enumerate(class_map)):
        
        num_ins=0
        path_dir=os.path.join(DATA_ROOT,'Audioset','Audioset_{}_01'.format(str(inst)))
        lim_ins=MAX_PM2_CLIPS-counts[str(inst)]
        with open(path_dir, "rb") as fp:   # Unpickling
            AS = pickle.load(fp)
            as_scores=compute_mean_max.get_scores_NN('NN_ALL',AS[0])
            for idx, score in enumerate(as_scores):
                if AS[2][idx][0] not in pmic_test_set:
                    if score[inst_idx]>=0.5 and num_ins < MAX_PM2_CLIPS:
                        PM2_X.append(AS[0][idx])
                        PM2_Y.append(str(inst))
                        num_ins=num_ins+1
                    elif score[20]>0.5 and num_sil < MAX_PM2_CLIPS:
                        PM2_X.append(AS[0][idx])
                        PM2_Y.append('zilence')
                        num_sil=num_sil+1         
    
    PM2_X=np.array(PM2_X)
    PM2_Y=np.array(PM2_Y)
    
    print('PM2 counts')
    print(collections.Counter(PM2_Y))
    with open(os.path.join(DATA_ROOT,'PM2'), "wb") as fp:   #Pickling
        pickle.dump([PM2_X,PM2_Y], fp,protocol=4)
    
    