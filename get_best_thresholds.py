# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 06:33:07 2019

@author: Gonçalo
"""
import numpy as np
import os
import pandas as pd

from sklearn.metrics import average_precision_score,  recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier

classes=['accordion' , 'banjo' , 'bass' , 'cello' , 'clarinet','cymbals', 'drums', 
         'flute', 'guitar','mallet_percussion' ,'mandolin','organ','piano',
         'saxophone','synthesizer','trombone','trumpet','ukulele','violin','voice','zillence']


DATA_ROOT='D:\CastelBranco\PAPER'
SCORES_NAME='TEST_SCORES'
THRESHS_NAME='threshs_test'
OPENMIC = np.load(os.path.join(DATA_ROOT,'OpenMIC', 'openmic-2018.npz'))
X_om_final, Y_true_om_final, Y_mask_om_final, sample_key_om_final = OPENMIC['X'], OPENMIC['Y_true'], OPENMIC['Y_mask'], OPENMIC['sample_key']

split_train = pd.read_csv(os.path.join(DATA_ROOT, 'OpenMIC','split01_train.csv'), 
                              header=None, squeeze=True)
split_test = pd.read_csv(os.path.join(DATA_ROOT, 'OpenMIC','split01_test.csv'), 
                     header=None, squeeze=True)


train_set = set(split_train)
test_set = set(split_test)

idx_train, idx_test = [], []

for idx, n in enumerate(sample_key_om_final):
    if n in train_set:
        idx_train.append(idx)
    elif n in test_set:
        idx_test.append(idx)

idx_train = np.asarray(idx_train)
idx_test = np.asarray(idx_test)

X_train = X_om_final[idx_train]
X_test = X_om_final[idx_test]

N_train = sample_key_om_final[idx_train]
N_test = sample_key_om_final[idx_test]


Y_true_train = Y_true_om_final[idx_train]
Y_true_test = Y_true_om_final[idx_test]

Y_mask_train = Y_mask_om_final[idx_train]
Y_mask_test = Y_mask_om_final[idx_test]



SCORES=np.load(os.path.join(DATA_ROOT,'scores', SCORES_NAME+'.npz'))['score']

np.arange(10)
tentativas=np.arange(0,1,0.01)

best_threshs=np.zeros(20)

SCORES=SCORES[idx_train]
for inst in range(20):
    print(classes[inst])

    best_ap_score=0
    for t in tentativas:    
        #precisions[inst]=t
        new_omic_y=[]
        for idx, score in enumerate(SCORES):
            labels=[0.5]*20
            #for instr in range(20):
            if Y_true_train[idx,inst] == 0.5:
                if score[inst] > t :#precisions[inst]:
                    labels[inst]=1
                        
        
            else:
                labels[inst]=Y_true_train[idx,inst]
            new_omic_y.append(labels)
        new_omic_y=np.array(new_omic_y) 
    
        # Map the instrument name to its column number
        
            
        
        ###TRAIN
        score_indexes=new_omic_y[:,inst] != 0.5       
        inst_y_train=new_omic_y[score_indexes,inst]
        inst_x_train=X_train[score_indexes]
    
        inst_binary_scores_train=np.zeros(len(inst_y_train))
        for idx,s in enumerate(inst_y_train):
            if s > 0.5:
                inst_binary_scores_train[idx]=1
        
        ###TEST
        score_indexes=Y_true_test[:,inst] != 0.5       
        inst_y_test=Y_true_test[score_indexes,inst]
        inst_x_test=X_test[score_indexes]
    
        inst_binary_scores_test=np.zeros(len(inst_y_test))
        for idx,s in enumerate(inst_y_test):
            if s > 0.5:
                inst_binary_scores_test[idx]=1        
                
    
        # Step 3: simplify the data by averaging over time
        
        # Let's arrange the data for a sklearn Random Forest model 
        # Instead of having time-varying features, we'll summarize each track by its mean feature vector over time
        X_train_ = np.mean(inst_x_train, axis=1)
        X_test_  = np.mean(inst_x_test, axis=1)
        
        #X_train=X_train.reshape(-1,1)
        #X_test=X_test.reshape(-1,1)
    
        # Step 3.
        # Initialize a new classifier
        clf = RandomForestClassifier(max_depth=8, n_estimators=100, random_state=0)
        
        # Step 4.
        clf.fit(X_train_, inst_binary_scores_train)
    
        # Step 5.
        # Finally, we'll evaluate the model on both train and test
        Y_pred_test = clf.predict(X_test_)
        

        prec_score=precision_score(inst_binary_scores_test, Y_pred_test)
        rec_score=recall_score(inst_binary_scores_test, Y_pred_test)
        ap_score=average_precision_score(inst_binary_scores_test, Y_pred_test)

        if ap_score > best_ap_score:
            best_ap_score=ap_score
            best_threshs[inst]=t
            prec=prec_score
            rec=rec_score
        
    #print('best_precision: {}, recall: {}, f: {}'.format(best_prec_score,rec,f))
    print('best_AP_score: {}, precision: {}, recall: {}'.format(best_ap_score,prec,rec))

np.savez(os.path.join(DATA_ROOT,'scores',THRESHS_NAME+'.npz') , threshs_up=best_threshs)
