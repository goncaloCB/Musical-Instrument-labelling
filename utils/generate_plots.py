# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 05:43:06 2019

@author: Gonçalo
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.metrics import precision_recall_curve

classes=['accordion' , 'banjo' , 'bass' , 'cello' , 'clarinet','cymbals', 'drums', 
         'flute', 'guitar','mallet_percussion' ,'mandolin','organ','piano',
         'saxophone','synthesizer','trombone','trumpet','ukulele','violin','voice','zillence']

DATA_ROOT='D:\CastelBranco\PAPER'
SCORES_NAME='TEST_SCORES'
THRESHS_NAMES_UP='TEST_THRESHS_UP'
THRESHS_NAMES_DOWN='TEST_THRESHS_DOWN'

SCORES=np.load(os.path.join(DATA_ROOT,'scores', SCORES_NAME+'.npz'))['score']


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


if __name__ == '__main__':
    #disable showing graphs
    precisions=[]
    recalls=[]
    matplotlib.use('Agg')
    for inst in range(20):  
    
        score_indexes=Y_true_test[:,inst] != 0.5       
        inst_scores_truth=Y_true_test[score_indexes,inst]
        inst_scores_pred=SCORES[idx_test][score_indexes,inst]
        inst_names=N_test[score_indexes]
    
        #c=test_mean_labels[score_indexes,inst]
        inst_binary_scores=np.zeros(len(inst_scores_truth))
        for idx,s in enumerate(inst_scores_truth):
            if s > 0.5:
                inst_binary_scores[idx]=1
        precision = dict()
        recall    = dict()
        thresh    = dict()
        
        precision, recall, thresh = precision_recall_curve(inst_binary_scores,inst_scores_pred)
        
        plt.figure()
        plt.plot(recall,precision)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.title('{} Precision-Recall curve'.format(classes[inst]))
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.grid(True)
        plt.savefig(os.path.join(DATA_ROOT,'plots','PR',classes[inst] + '_PR.png'))
        
        plt.figure()
        plt.plot(thresh,precision[0:-1],'red')
        plt.plot(thresh,recall[0:-1],'blue')
        plt.title('Precision-Recall per threshold')
        plt.ylabel('precision and recall')
        plt.xlabel('threshold')
        plt.grid(True)
        plt.savefig(os.path.join(DATA_ROOT,'plots','thresh',classes[inst] + '_PR.png'))
        
    
        for idx,n in enumerate(precision):
            if n > 0.98:
                precisions.append(thresh[idx-1])
                break
    
        for idx,n in enumerate(recall):
            if n < 0.98:
                recalls.append(thresh[idx-1])  
                break
            
    matplotlib.use('Qt5Agg')
    
    np.savez(os.path.join(DATA_ROOT,'scores',THRESHS_NAMES_UP+'.npz') , threshs_up=precisions)
    np.savez(os.path.join(DATA_ROOT,'scores',THRESHS_NAMES_DOWN+'.npz') , threshs_down=precisions)