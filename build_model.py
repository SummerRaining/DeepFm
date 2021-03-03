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

def deepfm_model(sparse_cols,dense_cols,sparse_max_len,
                 k=8,dropout=0.5,dense_num=128,num_layer=1):
    '''

    Parameters
    ----------
    sparse_cols : list
        所有离散特征的名称
    dense_cols : list
        所有连续特征的名称。
    sparse_max_len : list
        所有离散特征的基数

    Returns
    -------
    model:deepfm模型

    '''    
    #输入部分，分为sparse和dense部分。
    sparse_inputs = []
    for f in sparse_cols:
        _input = Input([1],name = f)
        sparse_inputs.append(_input)
        
    dense_inputs = []
    for f in dense_cols:
        _input = Input([1],name = f)
        dense_inputs.append(_input)
        
    #FM的一阶部分。离散特征使用1维embedding代替onehot和线性部分。
    sparse_1d_embed = []
    for i,_input in enumerate(sparse_inputs):
        _num = sparse_max_len[i]
        _embed = Flatten()(Embedding(_num+1,1,embeddings_regularizer=tf.keras.regularizers.l2(0.5))(_input))
        sparse_1d_embed.append(_embed)
    
    wx_sparse = Add()(sparse_1d_embed)
    wx_dense = Dense(1,name = 'first_order_dense')(Concatenate(axis = -1)(dense_inputs))
    wx = Add()([wx_sparse,wx_dense])
    
    #fm的二阶部分
    W = []
    #离散特征，embedding到k维，得到其隐向量。wi
    for i,_input in enumerate(sparse_inputs):
        _num = sparse_max_len[i]
        _embed = Flatten()(Embedding(_num+1, k, embeddings_regularizer=tf.keras.regularizers.l2(0.5))(_input))
        W.append(_embed)
        
    #连续特征。每个连续特征都映射到k维，而不是合起来到k维，且没有bias。dense全连接到k维，得到其隐向量。wi
    for f in dense_inputs:    
        _hidden = Dense(k, use_bias=False)(f)
        W.append(_hidden)
    
    #先相加再平方。
    frs_part = Add()(W)
    frs_part = Multiply()([frs_part,frs_part]) 
    #先平方再相加
    scd_part = Add()([Multiply()([_x,_x]) for _x in W])
    #相减，乘0.5.
    fm_part = Subtract()([frs_part,scd_part])
    fm_part = Lambda(lambda x:K.sum(x,axis = 1,keepdims = True)*0.5)(fm_part)
    
    #DNN部分。所有embedding concate在一起，然后训练。
    dnn = Concatenate()(W)
    for i in range(num_layer):
        dnn = Dropout(dropout)(Dense(dense_num,activation='relu')(dnn))
    dnn = Dense(1,activation='linear')(dnn)
    output = Activation('sigmoid')(Add()([wx,fm_part,dnn]))     #输出结果
    model = Model(sparse_inputs+dense_inputs,output)     #模型编译
    # plot_model(model, "deepfm.png")
    return model

def nfm_model(sparse_cols,dense_cols,sparse_max_len):
    '''

    Parameters
    ----------
    sparse_cols : list
        所有离散特征的名称
    dense_cols : list
        所有连续特征的名称。
    sparse_max_len : list
        所有离散特征的基数

    Returns
    -------
    model:deepfm模型

    '''
    #超参数
    k = 8
    
    #输入部分，分为sparse和dense部分。
    sparse_inputs = []
    for f in sparse_cols:
        _input = Input([1],name = f)
        sparse_inputs.append(_input)
        
    dense_inputs = []
    for f in dense_cols:
        _input = Input([1],name = f)
        dense_inputs.append(_input)
        
    #FM的一阶部分。离散特征使用1维embedding代替onehot和线性部分。
    sparse_1d_embed = []
    for i,_input in enumerate(sparse_inputs):
        _num = sparse_max_len[i]
        _embed = Flatten()(Embedding(_num+1,1,embeddings_regularizer=tf.keras.regularizers.l2(0.5))(_input))
        sparse_1d_embed.append(_embed)
    
    wx_sparse = Add()(sparse_1d_embed)
    wx_dense = Dense(1,name = 'first_order_dense')(Concatenate(axis = -1)(dense_inputs))
    wx = Add()([wx_sparse,wx_dense])
    
    #fm的二阶部分
    W = []
    #离散特征，embedding到k维，得到其隐向量。wi
    for i,_input in enumerate(sparse_inputs):
        _num = sparse_max_len[i]
        _embed = Flatten()(Embedding(_num+1, k, embeddings_regularizer=tf.keras.regularizers.l2(0.5))(_input))
        W.append(_embed)
        
    #连续特征。每个连续特征都映射到k维，而不是合起来到k维，且没有bias。dense全连接到k维，得到其隐向量。wi
    for f in dense_inputs:    
        _hidden = Dense(k, use_bias=False)(f)
        W.append(_hidden)
    
    #先相加再平方。
    frs_part = Add()(W)
    frs_part = Multiply()([frs_part,frs_part]) 
    #先平方再相加
    scd_part = Add()([Multiply()([_x,_x]) for _x in W])
    
    #相减，乘0.5.
    bi_interact_pool = Lambda(lambda x:x*0.5)(Subtract()([frs_part,scd_part]))
    fm_part = Lambda(lambda x:K.sum(x,axis = 1,keepdims = True))(bi_interact_pool)
    
    #DNN部分。所有embedding concate在一起，然后训练。
    W.append(bi_interact_pool)
    dnn = Concatenate()(W)
    dnn = Dropout(0.5)(Dense(128,activation='relu')(dnn))
    dnn = Dense(1,activation='linear')(dnn)
    output = Activation('sigmoid')(Add()([wx,fm_part,dnn]))
    model = Model(sparse_inputs+dense_inputs,output)
    # plot_model(model, "deepfm.png")
    return model



def autoint_model(sparse_cols,dense_cols,sparse_max_len):
    #超参数
    k = 8  #embedding的大小
    n_layer = 3
    d = 6 #attention中embedding的大小
    n_attention_head= 2
    #输入部分，分为sparse和dense部分。
    sparse_inputs = []
    for f in sparse_cols:
        _input = Input([1],name = f)
        sparse_inputs.append(_input)
        
    dense_inputs = []
    for f in dense_cols:
        _input = Input([1],name = f)
        dense_inputs.append(_input)
            
    W = []
    #离散特征，embedding到k维，得到其隐向量。wi
    for i,_input in enumerate(sparse_inputs):
        _num = sparse_max_len[i]
        _embed = Flatten()(Embedding(_num+1, k, embeddings_regularizer=tf.keras.regularizers.l2(0.5))(_input))
        W.append(_embed)
        
    #连续特征。每个连续特征都映射到k维，而不是合起来到k维，且没有bias。dense全连接到k维，得到其隐向量。wi
    for f in dense_inputs:    
        _hidden = Dense(k, use_bias=False)(f)
        W.append(_hidden)
    embed_x = Concatenate(axis = 1)([Reshape([1,-1])(x) for x in W])
    
    def multi_head_attention(x,d = 6,n_attention_head= 2):
        attention_heads = []
        for i in range(n_attention_head):
            embed_q= Dense(d,use_bias=False)(x)
            embed_k= Dense(d,use_bias=False)(x)
            embed_v= Dense(d,use_bias=False)(x) #batch*5*d.
            
            attention = tf.matmul(embed_q,tf.transpose(embed_k,[0,2,1])) #batch*5*5
            attention = tf.nn.softmax(attention) #batch*5*5
            
            attention_output = tf.matmul(attention,embed_v) #batch*5*d
            attention_heads.append(attention_output)
            
        multi_head_output = Concatenate(axis = -1)(attention_heads) #batch*5*(2d)
        
        #resnet
        w_res = Dense(d*n_attention_head,use_bias=False)(x) #batch,5,d*n_attention_head
        output = Activation('relu')(Add()([w_res,multi_head_output]))
        return output

    for i in range(n_layer):
        embed_x = multi_head_attention(embed_x,d,n_attention_head)
    autoint_layer = Reshape([-1])(embed_x)
    
    #输出层
    output_layer = Dense(1,activation = None)(autoint_layer)
    output_layer = Activation('sigmoid')(output_layer)
    model = Model(sparse_inputs+dense_inputs,output_layer)
    return model

    
    
    