# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 16:49:26 2019

@author: Gonçalo Castel-Branco
"""
import numpy as np
import os
import soundfile as sf
from tqdm import tqdm
import glob
import json
from openmic import vggish
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer


DATA_ROOT='D:\CastelBranco\PAPER'
AUDIO_PATH=os.path.join(DATA_ROOT,'PureMic_audio')

def get_melspec(folder):
    """
    comentar
    """
    
    mel_spec=[]
    clips_names=[]
    
    #process list é a lista dos ficheiros wav presentes na pasta
    process_list = glob.glob(folder+'\\*.wav')
    process_list = np.array(process_list)
    

    for idx, filename in enumerate(process_list):
        
        vggish.params.EXAMPLE_HOP_SECONDS=0.1
        data, sample_rate = sf.read(filename)
        #alterei vggish/inpouts.py -> waveform_to_examples para receber
        #o EXAMPLE_HOP_SECONDS como entrada que por defeito é 0.96
        # ou seja semsobreposição
        log_mel=vggish.waveform_to_examples(data,sample_rate)
        
                
        mel_spec.append(log_mel)
        clips_names.append(filename)
        
        
    
    return mel_spec,clips_names

def compute_embeddings(data):
    with tf.Graph().as_default(), tf.Session() as sess:
        time_points, features = vggish.model.transform(data, sess)
    return features


def compute_pca(data):
    post_proc=vggish.postprocessor
    pca_params_path =os.path.join(DATA_ROOT,'vggish', 'vggish_pca_params.npz')
    PostProcessor = post_proc.Postprocessor(pca_params_path)
    postprocess = PostProcessor.postprocess
    
    post = postprocess(data)
    
    return post


def chunks(data):
    lim=4000
    tam=len(data)
    n_batches=int(np.ceil(tam/lim))
    batches=[]
    for d in range(n_batches):
        if d == (n_batches-1):
            batches.append(data[d*lim:])
        else:
            batches.append(data[d*lim:d*lim+lim])
    return np.array(batches)             

        
if __name__ == '__main__':
    with open(os.path.join(DATA_ROOT, 'class-map.json'), 'r') as f:
        class_map = json.load(f)
    
    #change in ...\site-packages\openmic\vggish
    # the param vggish.params.EXAMPLE_HOP_SECONDS to 0.1 (100ms hop)

    insts_embeddings=[]
    clips_names=[]
    data_labels=[]
    
    for instrument in tqdm(class_map):    
        inst_mel,clip_name=get_melspec(os.path.join(AUDIO_PATH,str(instrument)))
        
        flattened=np.array(inst_mel).reshape(-1,96,64)
        if len(flattened) > 4000:
            batches=chunks(flattened)
        
        all_embeddings=[]
        for batch in batches:
            features=compute_embeddings(batch) ##flatten two first dimensions to feed vggish
            all_embeddings.append(features)   
            
        vggished=np.concatenate((all_embeddings),axis=0)
        pcaed=compute_pca(vggished)
        
        ######recover the flattened dimensions#################################
        offset=0

        for idx, framed_clip in enumerate(inst_mel):
            
            num_frames=len(framed_clip)
            
            insts_embeddings.append(pcaed[offset:offset+num_frames])
            data_labels.append(instrument)
            clips_names.append(os.path.splitext(os.path.basename(clip_name[idx]))[0])
            
            offset=offset+num_frames
            
        #######################################################################
        

    insts_embeddings=np.array(insts_embeddings)
    clips_names=np.array(clips_names)
    encoder = LabelBinarizer()
    y_true_binary = encoder.fit_transform(data_labels)
    np.savez(os.path.join(DATA_ROOT,"PureMic.npz") , X=insts_embeddings, Y=y_true_binary , sample_key=clips_names)
    