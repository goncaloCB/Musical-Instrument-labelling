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
THRESHS_NAMES_UP='threshs_test'
THRESHS_NAMES_DOWN='TEST_THRESHS_DOWN'

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
THRESHS_UP=np.load(os.path.join(DATA_ROOT,'scores', THRESHS_NAMES_UP+'.npz'))['threshs_up']
THRESHS_DOWN=np.load(os.path.join(DATA_ROOT,'scores', THRESHS_NAMES_DOWN+'.npz'))['threshs_down']

new_y_true_omic=[]
new_y_mask_omic=[]

count_pos=0
count_neg=0
for idx, score in enumerate(SCORES):
    labels=[0.5]*20
    masks=[False]*20
    for inst in range(20):
        if Y_true_om_final[idx,inst] == 0.5:
            if score[inst] > THRESHS_UP[inst]:
                #if sample_key_om_final[idx] in train_set:
                labels[inst]=1
                masks[inst]=True
                count_pos=count_pos+1
#            elif score[inst] < THRESHS_DOWN[inst]:
#                if sample_key_om_final[idx] in train_set:
#                    labels[inst]=0
#                    masks[inst]=True
#                    count_neg=count_neg+1
        else:
            labels[inst]=Y_true_om_final[idx,inst]
            masks[inst]=Y_mask_om_final[idx,inst]
    new_y_true_omic.append(labels)
    new_y_mask_omic.append(masks)
    
new_y_true_omic=np.array(new_y_true_omic) 
new_y_mask_omic=np.array(new_y_mask_omic) 



OPENMIC['X'], OPENMIC['Y_true'], OPENMIC['Y_mask'], OPENMIC['sample_key']
np.savez(os.path.join(DATA_ROOT,'OpenMIC','NEW_OpenMIC_FULL.npz') , X=X_om_final, Y_true=new_y_true_omic ,Y_mask=new_y_mask_omic ,sample_key=sample_key_om_final)
print('{} new positive labels for train set'.format(count_pos))
print('{} new negative labels for train set'.format(count_neg))