# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 09:13:55 2021

@author: shujie.wang
"""
import numpy as np
import pandas as pd
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
import os
from sklearn.metrics import roc_auc_score

if __name__ == '__main__':
    from get_data import get_data
    from build_model import deepfm_model,nfm_model,autoint_model
    
    x,y,x_val,y_val,x_test,y_test,sparse_max_len = get_data()
    sparse_max_len = [int(x) for x in sparse_max_len]
    
    sparse_cols = list(x.columns[:16])
    dense_cols = list(x.columns[16:])
    import tensorflow as tf
    model = deepfm_model(sparse_cols,dense_cols,sparse_max_len,k=4,dropout=0.9,dense_num=512,num_layer=1)
    # model = nfm_model(sparse_cols,dense_cols,sparse_max_len)
    # model = autoint_model(sparse_cols,dense_cols,sparse_max_len)
    model.compile(optimizer="adam", 
              loss="binary_crossentropy", 
              metrics=["binary_crossentropy", tf.keras.metrics.AUC(name='auc')])

    checkpoint_path = 'models/epoch_{epoch:03d}train_auc{auc:.3f}_val_auc{val_auc:.3f}.h5'
    checkpoint = ModelCheckpoint(
            checkpoint_path,monitor='val_auc',
            save_best_only=True,save_weights_only=False,verbose=1,mode='max')
    early_stopping = EarlyStopping(monitor='val_auc', min_delta=0, patience=10,mode='max', verbose=1)
    
    #准备数据
    train_x = [x[f].values.astype(int) for f in sparse_cols] + [x[f].values for f in dense_cols]
    val_x = [x_val[f].values.astype(int) for f in sparse_cols] + [x_val[f].values for f in dense_cols]
    model.fit(train_x, y, epochs=30, batch_size=256,
          validation_data=(val_x, y_val),
          callbacks=[checkpoint,early_stopping]
         )
    
    model = load_model('models/'+os.listdir('models')[-1])
    
    #预测模型的auc值。
    test_x = [x_test[f].values.astype(int) for f in sparse_cols] + [x_test[f].values for f in dense_cols]
    test_proba = model.predict(test_x)
    print(roc_auc_score(y_test,test_proba))
    
