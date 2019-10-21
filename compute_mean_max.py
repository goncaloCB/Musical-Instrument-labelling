
from tqdm import tqdm
import pickle
import os
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

DATA_ROOT='D:\CastelBranco\PAPER'
NUM_CLASSES=21


def get_indexes(unique_names,all_names):
    indexes=[]
    for name in tqdm(unique_names):
        indexes.append(name==all_names)
    
    return indexes


def get_Omic_data():
    path_dir=os.path.join(DATA_ROOT,'OpenMIC','OpenMic_01')
    with open(path_dir, "rb") as fp:   # Unpickling
        OM = pickle.load(fp)
        
    omic_x_all=[]
    omic_names_all=[]
    omic_y_all=[]
                
                
    for idx, clip in enumerate(OM[0]):
        for sec, instance in enumerate(clip):
            omic_x_all.append(instance)
            omic_y_all.append(OM[1][idx])
            omic_names_all.append(OM[2][idx])
    
    omic_x_all=np.array(omic_x_all)
    omic_y_all=np.array(omic_y_all)
    omic_names_all=np.array(omic_names_all)
    
    return omic_x_all, omic_y_all, omic_names_all, OM[2]

def get_Pmic_data():
    PUREMIC = np.load(os.path.join(DATA_ROOT, 'PureMic.npz'))

    X_PureMic, Y_PureMic, sample_key_PureMic = PUREMIC['X'], PUREMIC['Y'], PUREMIC['sample_key']
        

    X_PM_ALL=[]
    Y_PM_ALL=[]
    samp_info_all_pm=[]
    
    for idx, clip in enumerate(X_PureMic):
        for sec, instance in enumerate(clip):
            X_PM_ALL.append(instance)
            Y_PM_ALL.append(Y_PureMic[idx])
            samp_info_all_pm.append([sample_key_PureMic[idx],sec])
                
    X_PM_ALL=np.array(X_PM_ALL)
    Y_PM_ALL=np.array(Y_PM_ALL)
    samp_info_all_pm=np.array(samp_info_all_pm)
    
    
    
    return X_PM_ALL, Y_PM_ALL, samp_info_all_pm, sample_key_PureMic

def get_omic_splits(om_sample_names):

    split_train = pd.read_csv(os.path.join(DATA_ROOT, 'OpenMIC','split01_train.csv'), 
                              header=None, squeeze=True)
    split_test = pd.read_csv(os.path.join(DATA_ROOT, 'OpenMIC','split01_test.csv'), 
                         header=None, squeeze=True)

    train_set = set(split_train)
    test_set = set(split_test)
    
    idx_train, idx_test = [], []

    for idx, n in enumerate(om_sample_names):
        if n in train_set:
            idx_train.append(idx)
        elif n in test_set:
            idx_test.append(idx)
    
    idx_train = np.asarray(idx_train)
    idx_test = np.asarray(idx_test)
    
    return idx_train, idx_test


def compute_mean(indexes,scores):
    train_mean_labels=[]
    for ind in tqdm(indexes):
        labels_mean=np.zeros(21)
        for idx in range(NUM_CLASSES):
            labels_mean[idx]=scores[ind,idx].mean()
        
        train_mean_labels.append(labels_mean)

    return np.array(train_mean_labels)

def compute_max(indexes,scores):
    train_mean_labels=[]
    for ind in tqdm(indexes):
        labels_mean=np.zeros(21)
        for idx in range(NUM_CLASSES):
            labels_mean[idx]=scores[ind,idx].max()
        
        train_mean_labels.append(labels_mean)

    return np.array(train_mean_labels)

def get_scores_NN(model_name,data):
    img_input = Input(shape=(128,))
    n=Dense(4096, activation='sigmoid')(img_input)
    n=Dense(2048, activation='sigmoid')(n)
    n=Dense(NUM_CLASSES, activation='softmax')(n)
    NN_FINAL=Model(img_input,n)
    NN_FINAL.load_weights(os.path.join(DATA_ROOT,'models',model_name+'.h5'))
    
    return NN_FINAL.predict(data)

if __name__ == '__main__':
    
    SCORE_NAME='TEST_SCORES'
    all_X,_,all_N,N=get_Pmic_data()
    
    scores=get_scores_NN('NN_MEAN',all_X)
    
    indexes=get_indexes(N,all_N[:,0])
    
    mean_scores=compute_mean(indexes,scores)
    
    np.savez(os.path.join(DATA_ROOT,'scores',SCORE_NAME+'.npz') , score=mean_scores)
    
    
#    inst=0
#    for idx, n in enumerate(mean_scores):
#        if idx%50==0 and idx !=0:
#            inst=inst+1
#        if np.argmax(n) != inst:
#            print(N[idx])