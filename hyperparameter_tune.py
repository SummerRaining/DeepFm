# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 14:13:18 2021

@author: shujie.wang
"""
from get_data import get_data
from auto_int import autoint_model
from deepfm_model import deepfm_model
import tensorflow as tf
import itertools     
from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.models import load_model
    
if __name__ == '__main__':
    k = [4,8,16]
    dropout = [0.1,0.3,0.5,0.7,0.9]
    dense_num = [64,128,256,512]
    num_layer = [0,1,2,3,4]
    params = [k,dropout,dense_num,num_layer]
    params = list(itertools.product(*params))

    #
    x,y,x_val,y_val,x_test,y_test,sparse_max_len = get_data()
    sparse_max_len = [int(x) for x in sparse_max_len]
    sparse_cols = list(x.columns[:16])
    dense_cols = list(x.columns[16:])
    recoder = []
    for _param in params:
        checkpoint_path = 'models/best_model.h5'
        checkpoint = ModelCheckpoint(
            checkpoint_path,monitor='val_auc',
            save_best_only=True,save_weights_only=False,verbose=0,mode='max')
        early_stopping = EarlyStopping(monitor='val_auc', min_delta=0, patience=10,mode='max', verbose=1)    
        model = deepfm_model(sparse_cols,dense_cols,sparse_max_len,
                             k=_param[0],dropout=_param[1],dense_num=_param[2],num_layer=_param[3])
        model.compile(optimizer="adam", 
                  loss="binary_crossentropy", 
                  metrics=["binary_crossentropy", tf.keras.metrics.AUC(name='auc')])
       
        #准备数据
        train_x = [x[f].values.astype(int) for f in sparse_cols] + [x[f].values for f in dense_cols]
        val_x = [x_val[f].values.astype(int) for f in sparse_cols] + [x_val[f].values for f in dense_cols]
        model.fit(train_x, y, epochs=30, batch_size=256,
              validation_data=(val_x, y_val),
              callbacks=[checkpoint,early_stopping],
              verbose = 0
             )
        model = load_model('models/best_model.h5')
        #预测模型的auc值。
        test_x = [x_test[f].values.astype(int) for f in sparse_cols] + [x_test[f].values for f in dense_cols]
        test_proba = model.predict(test_x)
        print(roc_auc_score(y_test,test_proba))
        recoder.append([_param,roc_auc_score(y_test,test_proba)])
    
    import pandas as pd
    result = pd.DataFrame([list(x[0]) + [x[1]] for x in recoder],columns = ['K','dropout','dense_num','num_layer','value'])
    result.to_csv('result.csv')
    
    
