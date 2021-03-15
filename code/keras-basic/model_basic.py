#!/usr/local/bin/python3
# -*- coding: utf-8 -*- 

from header import *



def keras_basic(hparam):
    if hparam['elmo_signature']=='tokens':
        input_text = keras.Input(shape=(hparam['maxlen'],), dtype=tf.string)
    elif hparam['elmo_signature']=='default':
        input_text = keras.Input(shape=(1,), dtype=tf.string)

    x = ElmoEmbeddingLayer_v2(hparam['elmo_signature'], hparam['elmo_tag'], hparam['embedding_trainable'])(input_text)


    # maxpool = keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid')(x)
    # avgpool = keras.layers.AveragePooling1D(pool_size=2, strides=1, padding='valid')(x)
    # x = keras.layers.Flatten()(x)

    if hparam['elmo_tag'] == 'word_emb':
        x = GlobalMaxPooling1D()(x)

        # x = SpatialDropout1D(0.3)(x)
        # x = Bidirectional(GRU(100, return_sequences=True))(x)
        # avg_pool = GlobalAveragePooling1D()(x)
        # max_pool = GlobalMaxPooling1D()(x)
        # x = concatenate([avg_pool, max_pool])
    elif hparam['elmo_tag'] == 'default':
        x = Dense(32, activation='relu')(x)
    

    # indie1 = keras.layers.Dense(16, activation='relu')(flat_1)

    x = Dense(1000, activation='relu')(x)
    output1 = keras.layers.Dense(hparam['T'], activation='sigmoid' if hparam['isDataBinary'] else 'relu')(x)

    model = keras.Model(inputs=input_text, outputs=output1)
    # print the model summary
    print(model.summary())
    return model



def keras_basic_conv(hparam):
    if hparam['elmo_signature']=='tokens':
        input_text = keras.Input(shape=(hparam['maxlen'],), dtype=tf.string)
    elif hparam['elmo_signature']=='default':
        input_text = keras.Input(shape=(1,), dtype=tf.string)

    x = ElmoEmbeddingLayer_v2(hparam['elmo_signature'], hparam['elmo_tag'], hparam['embedding_trainable'])(input_text)


    # maxpool = keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid')(x)
    # avgpool = keras.layers.AveragePooling1D(pool_size=2, strides=1, padding='valid')(x)
    # x = keras.layers.Flatten()(x)

    if hparam['elmo_tag'] == 'word_emb':
        x = keras.layers.Conv1D(filters=250, kernel_size=2, strides=1, padding='same', activation='relu')(x)
        x = GlobalMaxPooling1D()(x)

        # x = SpatialDropout1D(0.3)(x)
        # x = Bidirectional(GRU(100, return_sequences=True))(x)
        # avg_pool = GlobalAveragePooling1D()(x)
        # max_pool = GlobalMaxPooling1D()(x)
        # x = concatenate([avg_pool, max_pool])
    elif hparam['elmo_tag'] == 'default':
        x = Dense(32, activation='relu')(x)
    

    # indie1 = keras.layers.Dense(16, activation='relu')(flat_1)

    x = Dense(250, activation='relu')(x)
    x = Dropout(0.2)(x)
    output1 = keras.layers.Dense(hparam['T'], activation='sigmoid' if hparam['isDataBinary'] else 'relu')(x)

    model = keras.Model(inputs=input_text, outputs=output1)
    # print the model summary
    print(model.summary())
    return model


def keras_basic_bert(hyper_params):
    input_text = keras.Input(shape=hyper_params['input_shape'], dtype='float', name='trainable_input')

    dense1 = Dense(512, activation=hyper_params['activation'], input_shape=(34, 1024,))(input_text)
    dense2 = Dense(256, activation=hyper_params['activation'], input_shape=(34, 512,))(dense1)
    flat = Flatten(name='flat')(dense2)
    hidden1 = Dense(2048, activation=hyper_params['activation'], name='hidden1')(flat)

    dense3 = [Dense(512, activation=hyper_params['activation'], name='dense3_'+labels[i])(hidden1) for i in range(hyper_params['T'])]
    # dense = [CustomizedDenseLayer(output_dim=32)(conc) for _ in range(hyper_params['T'])]  #? what is remove this layer completedly
    dense = [Dense(128, activation=hyper_params['activation'], name='dense_'+labels[i])(l) for i,l in enumerate(dense3)]  #? what is remove this layer completedly
    # dense = ''

    output = [ Dense(1, activation='sigmoid', name=labels[i])(lyer) for i,lyer in enumerate(dense)]
    # output = [ Dense(1, activation='sigmoid' if hyper_params['isDataBinary'] else hyper_params['activation'])(flat) for _ in range(hyper_params['T'])]

    model = keras.Model(inputs=input_text, outputs=output) #? Loss shift to a specific task during training?
    # print the model summary
    print(model.summary())
    return model
