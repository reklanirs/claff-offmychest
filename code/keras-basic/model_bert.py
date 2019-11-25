#!/usr/local/bin/python3
# -*- coding: utf-8 -*- 

from header import *
from bert_serving.client import BertClient

def BERT_Embedding(l):
    from bert_serving.client import BertClient
    bc = BertClient()
    return bc.encode( list(l) )

class BERTEmbeddingLayer_Wrong(Layer):
    def __init__(self, **kwargs):
        self.bc = BertClient()
        super(BERTEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        print('## BERTEmbeddingLayer input_shape: {}'.format(input_shape))
        super(BERTEmbeddingLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return self.bc.encode(x.numpy()).squeeze()

    def compute_output_shape(self, input_shape):
        print('## BERTEmbeddingLayer input_shape: {}'.format(input_shape))
        return (input_shape[0], 768)



def keras_bert(hyper_params):
    input_text = keras.Input(shape=hyper_params['input_shape'], dtype='float', name='trainable_input')

    if len(hyper_params['input_shape']) == 2:
        # conv = keras.layers.Conv1D(filters=512, kernel_size=10, strides=1, padding='same', activation=hyper_params['activation'])(input_text)
        flat = Bidirectional(LSTM(512))(input_text)
        # conv = Bidirectional(GRU(512, return_sequences=True))(input_text)

        # avg_pool = AveragePooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(conv)
        # max_pool = MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(conv)
        # x = concatenate([avg_pool, max_pool])
        # flat = Flatten(name='flat')(x)
    elif len(hyper_params['input_shape']) == 1:
        flat = Dense(512, activation=hyper_params['activation'], name='Dense')(input_text)

    hidden1 = Dense(512, activation=hyper_params['activation'], name='hidden1')(flat)

    # dense = [CustomizedDenseLayer(output_dim=32)(conc) for _ in range(hyper_params['T'])]  #? what is remove this layer completedly
    dense = [Dense(64, activation=hyper_params['activation'], name='dense_'+labels[i])(hidden1) for i in range(hyper_params['T'])]  #? what is remove this layer completedly
    # dense = ''

    output = [ Dense(1, activation='sigmoid', name=labels[i])(lyer) for i,lyer in enumerate(dense)]
    # output = [ Dense(1, activation='sigmoid' if hyper_params['isDataBinary'] else hyper_params['activation'])(flat) for _ in range(hyper_params['T'])]

    model = keras.Model(inputs=input_text, outputs=output) #? Loss shift to a specific task during training?
    # print the model summary
    print(model.summary())
    return model

def keras_bert_prelu(hyper_params):
    input_text = keras.Input(shape=hyper_params['input_shape'], dtype='float', name='trainable_input')

    if len(hyper_params['input_shape']) == 2:
        tmp = keras.layers.Conv1D(filters=512, kernel_size=10, strides=1, padding='same')(input_text)
        conv = keras.layers.PReLU(init='zero', weights=None)(tmp)
        # conv = Bidirectional(GRU(512, return_sequences=True))(input_text)

        avg_pool = AveragePooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(conv)
        max_pool = MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(conv)
        x = concatenate([avg_pool, max_pool])

        flat = Flatten(name='flat')(x)
    elif len(hyper_params['input_shape']) == 1:
        tmp = Dense(512, name='Dense')(input_text)
        flat = keras.layers.PReLU(init='zero', weights=None)(tmp)

    hidden1_tmp = Dense(512, name='hidden1')(flat)
    hidden1 = keras.layers.PReLU(init='zero', weights=None)(hidden1_tmp)

    # dense = [CustomizedDenseLayer(output_dim=32)(conc) for _ in range(hyper_params['T'])]  #? what is remove this layer completedly
    dense_tmp = [Dense(64, name='dense_'+labels[i])(hidden1) for i in range(hyper_params['T'])]  #? what is remove this layer completedly
    dense = [keras.layers.PReLU(init='zero', weights=None)(i) for i in dense_tmp]
    # dense = ''

    output = [ Dense(1, activation='sigmoid', name=labels[i])(lyer) for i,lyer in enumerate(dense)]
    # output = [ Dense(1, activation='sigmoid' if hyper_params['isDataBinary'] else hyper_params['activation'])(flat) for _ in range(hyper_params['T'])]

    model = keras.Model(inputs=input_text, outputs=output) #? Loss shift to a specific task during training?
    # print the model summary
    print(model.summary())
    return model


def keras_bert_1(hyper_params):
    input_text = keras.Input(shape=hyper_params['input_shape'], dtype='float')

    # hidden1 = Dense(512, activation=hyper_params['activation'])(input_text)

    # bgru = Bidirectional(GRU(512, return_sequences=True))(input_text)

    dense = Dense(128, activation=hyper_params['activation'])(input_text)

    # flat = Flatten()(dense)

    output = Dense(1, activation='sigmoid', name=labels[0])(dense)
 
    model = keras.Model(inputs=input_text, outputs=output) #? Loss shift to a specific task during training?
    # print the model summary
    print(model.summary())
    return model


def keras_bert_dual(hyper_params):
    input_text = keras.Input(shape=hyper_params['input_shape'], dtype='float', name='constant_input')
    if len(hyper_params['input_shape']) == 2:
        conv = keras.layers.Conv1D(filters=512, kernel_size=10, strides=1, padding='same', activation=hyper_params['activation'])(input_text)
        # conv = Bidirectional(GRU(512, return_sequences=True))(input_text)

        avg_pool = AveragePooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(conv)
        max_pool = MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(conv)
        x = concatenate([avg_pool, max_pool])

        flat = Flatten(name='flat')(x)
    elif len(hyper_params['input_shape']) == 1:
        flat = Dense(512, activation=hyper_params['activation'], name='Dense')(input_text)

    input_text2 = keras.Input(shape=hyper_params['input_shape'], dtype='float', name='trainable_input')
    if len(hyper_params['input_shape']) == 2:
        conv2 = keras.layers.Conv1D(filters=512, kernel_size=10, strides=1, padding='same', activation=hyper_params['activation'])(input_text2)
        # conv = Bidirectional(GRU(512, return_sequences=True))(input_text)

        avg_pool2 = AveragePooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(conv2)
        max_pool2 = MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(conv2)
        x2 = concatenate([avg_pool2, max_pool2])

        flat2 = Flatten(name='flat2')(x2)
    elif len(hyper_params['input_shape']) == 1:
        flat2 = Dense(512, activation=hyper_params['activation'], name='Dense2')(input_text2)

    conc = concatenate([flat, flat2])

    hidden1 = Dense(512, activation=hyper_params['activation'], name='hidden1')(conc)

    # dense = [CustomizedDenseLayer(output_dim=32)(conc) for _ in range(hyper_params['T'])]  #? what is remove this layer completedly
    dense = [Dense(64, activation=hyper_params['activation'], name='dense_'+labels[i])(hidden1) for i in range(hyper_params['T'])]  #? what is remove this layer completedly
    # dense = ''

    output = [ Dense(1, activation='sigmoid', name=labels[i])(lyer) for i,lyer in enumerate(dense)]
    # output = [ Dense(1, activation='sigmoid' if hyper_params['isDataBinary'] else hyper_params['activation'])(flat) for _ in range(hyper_params['T'])]

    model = keras.Model(inputs=[input_text, input_text2], outputs=output) #? Loss shift to a specific task during training?
    # print the model summary
    print(model.summary())
    return model