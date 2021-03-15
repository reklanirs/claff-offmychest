#!/usr/local/bin/python3
# -*- coding: utf-8 -*- 

from header import *

def keras_bert_stl(hyper_params):
    # input_text = [keras.Input(shape=hyper_params['input_shape'], dtype='float') for _ in range(hyper_params['T'])]
    input_text = keras.Input(shape=hyper_params['input_shape'], dtype='float', name='trainable_input')
    flat = input_text
    # conv = keras.layers.Conv1D(filters=512, kernel_size=10, strides=1, padding='same', activation=hyper_params['activation'])(input_text)

    dense = [Dense(128, activation=hyper_params['activation'])(flat)  for _ in range(hyper_params['T'])]

    output = [ Dense(1, activation='sigmoid', name=labels[i])(lyer) for i,lyer in enumerate(dense)]
    model = keras.Model(inputs=input_text, outputs=output) #? Loss shift to a specific task during training?
    return model


def keras_bert_bk(hyper_params):
    input_text = keras.Input(shape=hyper_params['input_shape'], dtype='float', name='trainable_input')
    # conv = keras.layers.Conv1D(filters=512, kernel_size=10, strides=1, padding='same', activation=hyper_params['activation'])(input_text)
    # flat = Bidirectional(LSTM(256))(conv)
    flat = input_text
    dense = [Dense(128, activation=hyper_params['activation'], name='dense_'+labels[i])(flat) for i in range(hyper_params['T'])]  #? what is remove this layer completedly
    output = [ Dense(1, activation='sigmoid', name=labels[i])(lyer) for i,lyer in enumerate(dense)]
    model = keras.Model(inputs=input_text, outputs=output) #? Loss shift to a specific task during training?
    return model

def keras_bert(hyper_params):
    input_text = keras.Input(shape=hyper_params['input_shape'], dtype='float', name='trainable_input')

    # BatchNormalization
    # bn = keras.layers.BatchNormalization(axis=-1)(input_text)

    # att = tf.compat.v1.keras.layers.Attention()(input_text)
    if len(hyper_params['input_shape']) == 2:
        conv = keras.layers.Conv1D(filters=512, kernel_size=10, strides=1, padding='same', activation=hyper_params['activation'])(input_text)
        # flat = Bidirectional(LSTM(256))(conv)
        # bn = keras.layers.BatchNormalization(axis=-1)(conv1)
        # conv = keras.layers.Conv1D(filters=256, kernel_size=10, strides=1, padding='same', activation=hyper_params['activation'])(bn)
        # dp = Dropout(hyper_params['drop_rate'])(conv)
        # Add bath normalization!
        # Add different filter conv1D layer
        # Add Dropout 0.2
        # flat = Bidirectional(LSTM(512))(input_text)
        # conv = Bidirectional(GRU(512, return_sequences=True))(input_text)
        # bn2 = keras.layers.BatchNormalization(axis=-1)(conv)
        # attention
        # attention_pre = Dense(1, name='attention_vec')(conv)   #[b_size,maxlen,1]
        # attention_probs  = Softmax()(attention_pre)  #[b_size,maxlen,1] 
        # attention_mul = Lambda(lambda x:x[0]*x[1])([attention_probs,conv])

        avg_pool = AveragePooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(conv)
        max_pool = MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(conv)
        x = concatenate([avg_pool, max_pool])
        flat = Flatten(name='flat')(blstm)
    elif len(hyper_params['input_shape']) == 1:
        flat = Dense(512, activation=hyper_params['activation'], name='Dense')(input_text)

    conv = keras.layers.Conv1D(filters=512, kernel_size=10, strides=1, padding='same', activation=hyper_params['activation'])(input_text)
    
    # blstm = Bidirectional(LSTM(256))(conv)
    # flat1 = Flatten(name='flat')(conv)
    # flat2 = Dense(512, activation=hyper_params['activation'])(flat1)
    # flat = concatenate([blstm, flat2])
    # flat = Bidirectional(LSTM(256))(conv)

    mh = MultiHead(keras.layers.LSTM(units=256), layer_num=5, name='Multi-LSTMs')(conv)
    flat = keras.layers.Flatten(name='Flatten')(mh)


    # dense = [CustomizedDenseLayer(output_dim=32)(conc) for _ in range(hyper_params['T'])]  #? what is remove this layer completedly
    dense = [Dense(128, activation=hyper_params['activation'], name='dense_'+labels[i])(flat) for i in range(hyper_params['T'])]  #? what is remove this layer completedly
    # dense = ''

    output = [ Dense(1, activation='sigmoid', name=labels[i])(lyer) for i,lyer in enumerate(dense)]
    # output = [ Dense(1, activation='sigmoid' if hyper_params['isDataBinary'] else hyper_params['activation'])(flat) for _ in range(hyper_params['T'])]

    model = keras.Model(inputs=input_text, outputs=output) #? Loss shift to a specific task during training?
    # print the model summary
    print(model.summary())
    return model

def keras_bert_group1(hyper_params):
    '''
    Group1: Information_disclosure, Emotional_disclosure
    Group2: General_support
    Group3: Support, Info_support, Emo_support
    labels = ['Emotional_disclosure','Information_disclosure','Support','General_support','Info_support','Emo_support']
    '''
    assert(len(labels) == 6)
    input_text = keras.Input(shape=hyper_params['input_shape'], dtype='float', name='trainable_input')
    # conv = keras.layers.Conv1D(filters=512, kernel_size=10, strides=1, padding='same', activation=hyper_params['activation'])(input_text)
    # flat = Bidirectional(LSTM(256))(conv)

    if len(hyper_params['input_shape']) == 2:
        conv = keras.layers.Conv1D(filters=512, kernel_size=10, strides=1, padding='same', activation=hyper_params['activation'])(input_text)

        avg_pool = AveragePooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(conv)
        max_pool = MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(conv)
        x = concatenate([avg_pool, max_pool])
        flat = Flatten(name='flat')(x)
    elif len(hyper_params['input_shape']) == 1:
        flat = Dense(512, activation=hyper_params['activation'], name='Dense')(input_text)

    # hidden1 = Dense(512, activation=hyper_params['activation'], name='hidden1')(input_text)
    # dp = Dropout(hyper_params['drop_rate'])(hidden1)
    # flat = Dense(512, activation=hyper_params['activation'], name='hidden2')(dp)

    Group1 = Dense(256, activation=hyper_params['activation'], name='Group1')(flat)
    Group2 = Dense(256, activation=hyper_params['activation'], name='Group2')(flat)
    Group3 = Dense(256, activation=hyper_params['activation'], name='Group3')(flat)

    dense1 = [ Dense(128, activation=hyper_params['activation'], name='dense_'+labels[i])(Group1) for i in [0,1]]
    dense2 = [ Dense(128, activation=hyper_params['activation'], name='dense_'+labels[i])(Group2) for i in [3]]
    dense3 = [ Dense(128, activation=hyper_params['activation'], name='dense_'+labels[i])(Group3) for i in [2,4,5]]

    output1 = [ Dense(1, activation='sigmoid', name=labels[i])(lyer) for i,lyer in zip([0,1], dense1) ]
    output2 = [ Dense(1, activation='sigmoid', name=labels[i])(lyer) for i,lyer in zip([3], dense2) ]
    output3 = [ Dense(1, activation='sigmoid', name=labels[i])(lyer) for i,lyer in zip([2,4,5], dense3) ]

    model = keras.Model(inputs=input_text, outputs=output1 + [output3[0]] + output2 + output3[1:]) #? Loss shift to a specific task during training?
    return model


def keras_bert_group2(hyper_params):
    '''
    labels = ['Emotional_disclosure','Information_disclosure','Support','General_support','Info_support','Emo_support']

    Group1: Information_disclosure, Emotional_disclosure
    Group2: Information_disclosure, Support, Info_support
    Group3: Emotional_disclosure, Emo_support
    Group4: Support, General_support, Info_support, Emo_support
    Group5: General_support, Emo_support
    '''
    assert(len(labels) == 6)
    input_text = keras.Input(shape=hyper_params['input_shape'], dtype='float', name='trainable_input')
    conv = keras.layers.Conv1D(filters=512, kernel_size=10, strides=1, padding='same', activation=hyper_params['activation'])(input_text)
    flat = Bidirectional(LSTM(256))(conv)

    # hidden1 = Dense(512, activation=hyper_params['activation'], name='hidden1')(input_text)
    # dp = Dropout(hyper_params['drop_rate'])(hidden1)
    # flat = Dense(512, activation=hyper_params['activation'], name='hidden2')(dp)

    group1 = Dense(256, activation=hyper_params['activation'], name='Group1')(flat)
    group2 = Dense(256, activation=hyper_params['activation'], name='Group2')(flat)
    group3 = Dense(256, activation=hyper_params['activation'], name='Group3')(flat)
    group4 = Dense(256, activation=hyper_params['activation'], name='Group4')(flat)
    group5 = Dense(256, activation=hyper_params['activation'], name='Group5')(flat)

    # dense1 = [ Dense(128, activation=hyper_params['activation'], name='dense_'+labels[i])(Group1) for i in [0,1]]
    # dense2 = [ Dense(128, activation=hyper_params['activation'], name='dense_'+labels[i])(Group2) for i in [3]]
    # dense3 = [ Dense(128, activation=hyper_params['activation'], name='dense_'+labels[i])(Group3) for i in [2,4,5]]
    # hidden_layer = Dense(128, activation=hyper_params['activation'])

    dense1 = Dense(128, activation=hyper_params['activation'])(concatenate([group1, group2]))
    dense2 = Dense(128, activation=hyper_params['activation'])(concatenate([group1, group3]))
    dense3 = Dense(128, activation=hyper_params['activation'])(concatenate([group2, group4]))
    dense4 = Dense(128, activation=hyper_params['activation'])(concatenate([group4, group5]))
    dense5 = Dense(128, activation=hyper_params['activation'])(concatenate([group2, group4]))
    dense6 = Dense(128, activation=hyper_params['activation'])(concatenate([group3, group4, group5]))

    stack = [dense1,dense2,dense3,dense4,dense5,dense6]
    # flat = [Flatten()(lyer) for lyer in stack]

    # output1 = [ Dense(1, activation='sigmoid', name=labels[i])(lyer) for i,lyer in zip([0,1], dense1) ]
    # output2 = [ Dense(1, activation='sigmoid', name=labels[i])(lyer) for i,lyer in zip([3], dense2) ]
    # output3 = [ Dense(1, activation='sigmoid', name=labels[i])(lyer) for i,lyer in zip([2,4,5], dense3) ]
    output = [ Dense(1, activation='sigmoid', name=labels[i])(lyer) for i,lyer in enumerate(stack) ]

    model = keras.Model(inputs=input_text, outputs=output) #? Loss shift to a specific task during training?
    return model


def keras_bert_group_basic(hyper_params):
    '''
    Group1: Information_disclosure, Info_support
    Group2: Emotional_disclosure, Emo_support
    Group3: Support, General_support
    labels = ['Emotional_disclosure','Information_disclosure','Support','General_support','Info_support','Emo_support']
    '''
    assert(len(labels) == 6)
    input_text = keras.Input(shape=hyper_params['input_shape'], dtype='float', name='trainable_input')

    flat = Dense(1024, activation=hyper_params['activation'], name='hidden1')(input_text)

    sep = [Dense(512, activation=hyper_params['activation'])(flat) for i in range(len(labels))]
    sep2 = [Dense(128, activation=hyper_params['activation'])(sep[i]) for i in range(len(labels))]

    output = [Dense(1, activation='sigmoid', name=labels[i])(sep2[i]) for i in range(len(labels))]

    model = keras.Model(inputs=input_text, outputs=output) #? Loss shift to a specific task during training?
    return model


def keras_bert_group_ff(hyper_params):
    '''
    Group1: Information_disclosure, Info_support
    Group2: Emotional_disclosure, Emo_support
    Group3: Support, General_support
    labels = ['Emotional_disclosure','Information_disclosure','Support','General_support','Info_support','Emo_support']
    '''
    assert(len(labels) == 6)
    input_text = keras.Input(shape=hyper_params['input_shape'], dtype='float', name='trainable_input')
    # conv = keras.layers.Conv1D(filters=512, kernel_size=10, strides=1, padding='same', activation=hyper_params['activation'])(input_text)
    # conv = Dropout(hyper_params['drop_rate'])(conv)
    # hidden1 = Bidirectional(LSTM(256, return_sequences=False))(conv)
    # hidden1 = BatchNormalization(axis=-1)(hidden1)
    hidden1 = input_text


    #distribution layers
    # d = [Dense(distribution_nodes, activation=hyper_params['activation'])(hidden1) for i in range(hyper_params['T'])]
    base = 2048
    double_nodes = base*2
    triple_nodes = base*3

    infod_s_infos = Dense(triple_nodes, activation=hyper_params['activation'])(hidden1)
    s_gs_emos = Dense(triple_nodes, activation=hyper_params['activation'])(hidden1)

    infod_s = Dense(double_nodes, activation=hyper_params['activation'])(infod_s_infos)
    infod_infos = Dense(double_nodes, activation=hyper_params['activation'])(infod_s_infos)
    s_infos = Dense(double_nodes, activation=hyper_params['activation'])(infod_s_infos)

    s_gs = Dense(double_nodes, activation=hyper_params['activation'])(s_gs_emos)
    s_emos = Dense(double_nodes, activation=hyper_params['activation'])(s_gs_emos)
    gs_emos = Dense(double_nodes, activation=hyper_params['activation'])(s_gs_emos)

    infod_emod = Dense(double_nodes, activation=hyper_params['activation'])(hidden1)
    emod_emos = Dense(double_nodes, activation=hyper_params['activation'])(hidden1)


    #task_layers
    task_nodes = base
    infod = Dense(task_nodes, activation=hyper_params['activation'])(concatenate([infod_s, infod_infos, hidden1]))
    attention_probs1 = Softmax()(Dense(task_nodes)(infod))
    infod = Lambda(lambda x:x[0]*x[1])([attention_probs1,infod])

    emod = Dense(task_nodes, activation=hyper_params['activation'])(concatenate([infod_emod, emod_emos, hidden1]))
    attention_probs2 = Softmax()(Dense(task_nodes)(emod))
    emod = Lambda(lambda x:x[0]*x[1])([attention_probs2,emod])

    s = Dense(task_nodes, activation=hyper_params['activation'])(concatenate([s_infos, s_gs, s_emos, gs_emos, hidden1]))
    attention_probs3 = Softmax()(Dense(task_nodes)(s))
    s = Lambda(lambda x:x[0]*x[1])([attention_probs3,s])

    gs = Dense(task_nodes, activation=hyper_params['activation'])(concatenate([s_gs, gs_emos, hidden1]))
    attention_probs4 = Softmax()(Dense(task_nodes)(gs))
    gs = Lambda(lambda x:x[0]*x[1])([attention_probs4,gs])

    infos = Dense(task_nodes, activation=hyper_params['activation'])(concatenate([infod_infos, s_infos, hidden1]))
    attention_probs5 = Softmax()(Dense(task_nodes)(infos))
    infos = Lambda(lambda x:x[0]*x[1])([attention_probs5,infos])

    emos = Dense(task_nodes, activation=hyper_params['activation'])(concatenate([s_emos, gs_emos, emod_emos, hidden1]))
    attention_probs6 = Softmax()(Dense(task_nodes)(emos))
    emos = Lambda(lambda x:x[0]*x[1])([attention_probs6,emos])

    stack = [infod, emod, s, gs, infos, emos]
    # mid = [Dense(512, activation=hyper_params['activation'])(lyer) for i,lyer in enumerate(stack)]
    # stack = [Flatten()(i) for i in stack]

    output = [Dense(1, activation='sigmoid', name=labels[i], kernel_constraint=max_norm(2.))(lyer) for i,lyer in enumerate(stack)]

    model = keras.Model(inputs=input_text, outputs=output) #? Loss shift to a specific task during training?
    return model



def keras_bert_group_ff2(hyper_params):
    '''
    Group1: Information_disclosure, Info_support
    Group2: Emotional_disclosure, Emo_support
    Group3: Support, General_support
    labels = ['Emotional_disclosure','Information_disclosure','Support','General_support','Info_support','Emo_support']
    '''
    assert(len(labels) == 6)
    input_text = keras.Input(shape=hyper_params['input_shape'], dtype='float', name='trainable_input')
    # conv = keras.layers.Conv1D(filters=512, kernel_size=10, strides=1, padding='same', activation=hyper_params['activation'])(input_text)
    # conv = Dropout(hyper_params['drop_rate'])(conv)
    # hidden1 = Bidirectional(LSTM(256, return_sequences=False))(conv)
    # hidden1 = BatchNormalization(axis=-1)(hidden1)
    hidden1 = input_text # 1024


    #distribution layers
    # d = [Dense(distribution_nodes, activation=hyper_params['activation'])(hidden1) for i in range(hyper_params['T'])]
    base = 512
    double_nodes = base*2
    triple_nodes = base*3

    infod_s_infos = Dense(triple_nodes, activation=hyper_params['activation'])(hidden1) # triple_nodes
    s_gs_emos = Dense(triple_nodes, activation=hyper_params['activation'])(hidden1)

    # Lambda(lambda x:x[0]*x[1])([Softmax()(Dense(triple_nodes)(infod_s_infos)),infod_s_infos])

    infod_s = Dense(double_nodes, activation=hyper_params['activation'])(Lambda(lambda x:x[0]*x[1])([Softmax()(Dense(triple_nodes)(infod_s_infos)),infod_s_infos]))
    infod_infos = Dense(double_nodes, activation=hyper_params['activation'])(Lambda(lambda x:x[0]*x[1])([Softmax()(Dense(triple_nodes)(infod_s_infos)),infod_s_infos]))
    s_infos = Dense(double_nodes, activation=hyper_params['activation'])(Lambda(lambda x:x[0]*x[1])([Softmax()(Dense(triple_nodes)(infod_s_infos)),infod_s_infos]))

    # Lambda(lambda x:x[0]*x[1])([Softmax()(Dense(triple_nodes)(s_gs_emos)),s_gs_emos])
    s_gs = Dense(double_nodes, activation=hyper_params['activation'])(Lambda(lambda x:x[0]*x[1])([Softmax()(Dense(triple_nodes)(s_gs_emos)),s_gs_emos]))
    s_emos = Dense(double_nodes, activation=hyper_params['activation'])(Lambda(lambda x:x[0]*x[1])([Softmax()(Dense(triple_nodes)(s_gs_emos)),s_gs_emos]))
    gs_emos = Dense(double_nodes, activation=hyper_params['activation'])(Lambda(lambda x:x[0]*x[1])([Softmax()(Dense(triple_nodes)(s_gs_emos)),s_gs_emos]))

    infod_emod = Dense(double_nodes, activation=hyper_params['activation'])(hidden1)
    emod_emos = Dense(double_nodes, activation=hyper_params['activation'])(hidden1)


    #task_layers
    task_nodes = base
    conc1 = concatenate([infod_s, infod_infos, hidden1])
    infod = Dense(task_nodes, activation=hyper_params['activation'])(Lambda(lambda x:x[0]*x[1])([Softmax()(Dense(6*base)(conc1)),conc1]))

    conc2 = concatenate([infod_emod, emod_emos, hidden1])
    emod = Dense(task_nodes, activation=hyper_params['activation'])(Lambda(lambda x:x[0]*x[1])([Softmax()(Dense(6*base)(conc2)),conc2]))

    conc3 = concatenate([s_infos, s_gs, s_emos, gs_emos, hidden1])
    s = Dense(task_nodes, activation=hyper_params['activation'])(Lambda(lambda x:x[0]*x[1])([Softmax()(Dense(10*base)(conc3)),conc3]))

    conc4 = concatenate([s_gs, gs_emos, hidden1])
    gs = Dense(task_nodes, activation=hyper_params['activation'])(Lambda(lambda x:x[0]*x[1])([Softmax()(Dense(6*base)(conc4)),conc4]))

    conc5 = concatenate([infod_infos, s_infos, hidden1])
    infos = Dense(task_nodes, activation=hyper_params['activation'])(Lambda(lambda x:x[0]*x[1])([Softmax()(Dense(6*base)(conc5)),conc5]))

    conc6 = concatenate([s_emos, gs_emos, emod_emos, hidden1])
    emos = Dense(task_nodes, activation=hyper_params['activation'])(Lambda(lambda x:x[0]*x[1])([Softmax()(Dense(8*base)(conc6)),conc6]))


    stack = [infod, emod, s, gs, infos, emos]
    # mid = [Dense(512, activation=hyper_params['activation'])(lyer) for i,lyer in enumerate(stack)]
    # stack = [Flatten()(i) for i in stack]

    output = [Dense(1, activation='sigmoid', name=labels[i], kernel_constraint=max_norm(2.))(lyer) for i,lyer in enumerate(stack)]

    model = keras.Model(inputs=input_text, outputs=output) #? Loss shift to a specific task during training?
    return model



def keras_bert_group_test(hyper_params):
    input_text = keras.Input(shape=hyper_params['input_shape'], dtype='float', name='trainable_input')
    flat = Dense(sum([label_set[i] for i in labels]), activation=hyper_params['activation'])(input_text)
    # stack = [Dense(128, activation=hyper_params['activation'])(MultiHeadAttention(head_num=4)(input_text)) for i in range(6)]
    # dense = [Dense(label_set[labels[i]], activation=hyper_params['activation'], name='dense_'+labels[i])(flat) for i in range(hyper_params['T'])]  #? what is remove this layer completedly
    tmp = [1928,2020, 2871]
    dense = [Dense(tmp[i], activation=hyper_params['activation'])(flat) for i in range(3)]
    emod = Dense(1, activation='sigmoid', name='Emotional_disclosure')(concatenate([dense[0], dense[1]]))
    infod = Dense(1, activation='sigmoid', name='Information_disclosure')(concatenate([dense[1], dense[2]]))
    # output = [ Dense(1, activation='sigmoid', name=labels[i])(lyer) for i,lyer in enumerate(dense)]
    model = keras.Model(inputs=input_text, outputs=[emod, infod]) #? Loss shift to a specific task during training?
    return model


    
def keras_bert_group_venn(hyper_params):
    input_text = keras.Input(shape=hyper_params['input_shape'], dtype='float', name='trainable_input')
    base = 4
    flat = Dense(2048*base, activation=hyper_params['activation'])(input_text)
    flat = input_text
    # stack = [Dense(128, activation=hyper_params['activation'])(MultiHeadAttention(head_num=4)(input_text)) for i in range(6)]
    # dense = [Dense(label_set[labels[i]], activation=hyper_params['activation'], name='dense_'+labels[i])(flat) for i in range(hyper_params['T'])]  #? what is remove this layer completedly
    d114 = Dense(round(114*base), activation=hyper_params['activation'])(flat)
    d36 = Dense(round(36*base), activation=hyper_params['activation'])(flat)
    d67 = Dense(round(67*base), activation=hyper_params['activation'])(flat)
    d190 = Dense(round(190*base), activation=hyper_params['activation'])(flat)
    d21 = Dense(round(21*base), activation=hyper_params['activation'])(flat)
    d88 = Dense(round(88*base), activation=hyper_params['activation'])(flat)
    d156 = Dense(round(156*base), activation=hyper_params['activation'])(flat)
    d1347 = Dense(round(1347*base), activation=hyper_params['activation'])(flat)
    d42 = Dense(round(42*base), activation=hyper_params['activation'])(flat)
    d11 = Dense(round(11*base), activation=hyper_params['activation'])(flat)
    d14 = Dense(round(14*base), activation=hyper_params['activation'])(flat)
    d12 = Dense(round(12*base), activation=hyper_params['activation'])(flat)
    d115 = Dense(round(115*base), activation=hyper_params['activation'])(flat)
    d104 = Dense(round(104*base), activation=hyper_params['activation'])(flat)
    d1625 = Dense(round(1625*base), activation=hyper_params['activation'])(flat)
    d2491 = Dense(round(2491*base), activation=hyper_params['activation'])(flat)
    d63 = Dense(round(63*base), activation=hyper_params['activation'])(flat)
    d131 = Dense(round(131*base), activation=hyper_params['activation'])(flat)
    d28 = Dense(round(28*base), activation=hyper_params['activation'])(flat)
    d38 = Dense(round(38*base), activation=hyper_params['activation'])(flat)
    d304 = Dense(round(304*base), activation=hyper_params['activation'])(flat)
    d27 = Dense(round(27*base), activation=hyper_params['activation'])(flat)
    d17 = Dense(round(17*base), activation=hyper_params['activation'])(flat)
    d12_2 = Dense(round(12*base), activation=hyper_params['activation'])(flat)
    d20 = Dense(round(20*base), activation=hyper_params['activation'])(flat)
    d55 = Dense(round(55*base), activation=hyper_params['activation'])(flat)
    d49 = Dense(round(49*base), activation=hyper_params['activation'])(flat)
    d25 = Dense(round(25*base), activation=hyper_params['activation'])(flat)
    d768 = Dense(round(768*base), activation=hyper_params['activation'])(flat)
    d356 = Dense(round(356*base), activation=hyper_params['activation'])(flat)
    d337 = Dense(round(337*base), activation=hyper_params['activation'])(flat)

    d_base = [Dense(1, activation=hyper_params['activation'], name='d_base{}'.format(i))(flat) for i in range(len(labels))]

    Emotional_disclosure = concatenate([ d1347,d156,d88,d21,d190,d42,d67,d114,d36,d11,d14,d104,d38,d63,d1625, d_base[0] ])
    Information_disclosure = concatenate([ d2491,d1625,d115,d12,d14,d104,d63,d131,d11,d38,d28,d12_2,d20,d36,d55,d114, d_base[1] ])
    Support = concatenate([ d337,d356,d25,d768,d55,d114,d20,d49,d12_2,d17,d27,d304,d28,d36,d67,d190,d21,d88,d156,d42,d11,d14,d12,d115,d104,d63,d38,d131, d_base[2] ])
    General_support = concatenate([ d42,d67,d36,d11,d20,d12_2,d49,d17,d27,d304,d28,d38, d_base[3] ])
    Info_support = concatenate([ d115,d12,d104,d14,d21,d88,d11,d12_2,d17,d27,d25,d768, d_base[4] ])
    Emo_support = concatenate([ d67,d190,d21,d12,d14,d11,d36,d114,d55,d20,d12_2,d17,d49,d25,d356, d_base[5]  ])

    stack = [Emotional_disclosure,Information_disclosure,Support,General_support,Info_support,Emo_support]


    # stack = []
    # for i,lyer in enumerate(stack1):
    #     attention_probs1 = Softmax()(Dense(label_len_shrinked[i]*base + 1024, activation=hyper_params['activation'])(lyer))
    #     tmp = Lambda(lambda x:x[0]*x[1])([attention_probs1,lyer])
    #     stack.append(tmp)
    stack = [Dense(512, activation=hyper_params['activation'])(lyer) for i,lyer in enumerate(stack)]
    output = [Dense(1, activation='sigmoid', name=labels[i])(lyer) for i,lyer in enumerate(stack)]

    model = keras.Model(inputs=input_text, outputs=output) #? Loss shift to a specific task during training?
    return model


def keras_bert_group_venn2(hyper_params):
    input_text = keras.Input(shape=hyper_params['input_shape'], dtype='float', name='trainable_input')
    base = 1
    # flat = Bidirectional(LSTM(256, return_sequences=True))(input_text)
    flat = Dense(ceil(1024*base), activation=hyper_params['activation'])(input_text)
    # stack = [Dense(128, activation=hyper_params['activation'])(MultiHeadAttention(head_num=4)(input_text)) for i in range(6)]
    # dense = [Dense(label_set[labels[i]], activation=hyper_params['activation'], name='dense_'+labels[i])(flat) for i in range(hyper_params['T'])]  #? what is remove this layer completedly
    d114 = Dense(ceil(114*base), activation=hyper_params['activation'])(flat)
    d36 = Dense(ceil(36*base), activation=hyper_params['activation'])(flat)
    d67 = Dense(ceil(67*base), activation=hyper_params['activation'])(flat)
    d190 = Dense(ceil(190*base), activation=hyper_params['activation'])(flat)
    d21 = Dense(ceil(21*base), activation=hyper_params['activation'])(flat)
    d88 = Dense(ceil(88*base), activation=hyper_params['activation'])(flat)
    d156 = Dense(ceil(156*base), activation=hyper_params['activation'])(flat)
    d1347 = Dense(ceil(1347*base), activation=hyper_params['activation'])(flat)
    d42 = Dense(ceil(42*base), activation=hyper_params['activation'])(flat)
    d11 = Dense(ceil(11*base), activation=hyper_params['activation'])(flat)
    d14 = Dense(ceil(14*base), activation=hyper_params['activation'])(flat)
    d12 = Dense(ceil(12*base), activation=hyper_params['activation'])(flat)
    d115 = Dense(ceil(115*base), activation=hyper_params['activation'])(flat)
    d104 = Dense(ceil(104*base), activation=hyper_params['activation'])(flat)
    d1625 = Dense(ceil(1625*base), activation=hyper_params['activation'])(flat)
    d2491 = Dense(ceil(2491*base), activation=hyper_params['activation'])(flat)
    d63 = Dense(ceil(63*base), activation=hyper_params['activation'])(flat)
    d131 = Dense(ceil(131*base), activation=hyper_params['activation'])(flat)
    d28 = Dense(ceil(28*base), activation=hyper_params['activation'])(flat)
    d38 = Dense(ceil(38*base), activation=hyper_params['activation'])(flat)
    d304 = Dense(ceil(304*base), activation=hyper_params['activation'])(flat)
    d27 = Dense(ceil(27*base), activation=hyper_params['activation'])(flat)
    d17 = Dense(ceil(17*base), activation=hyper_params['activation'])(flat)
    d12_2 = Dense(ceil(12*base), activation=hyper_params['activation'])(flat)
    d20 = Dense(ceil(20*base), activation=hyper_params['activation'])(flat)
    d55 = Dense(ceil(55*base), activation=hyper_params['activation'])(flat)
    d49 = Dense(ceil(49*base), activation=hyper_params['activation'])(flat)
    d25 = Dense(ceil(25*base), activation=hyper_params['activation'])(flat)
    d768 = Dense(ceil(768*base), activation=hyper_params['activation'])(flat)
    d356 = Dense(ceil(356*base), activation=hyper_params['activation'])(flat)
    d337 = Dense(ceil(337*base), activation=hyper_params['activation'])(flat)

    d_base = [Dense(1, activation=hyper_params['activation'], name='d_base{}'.format(i))(flat) for i in range(len(labels))]

    Emotional_disclosure = concatenate([ d1347,d156,d88,d21,d190,d42,d67,d114,d36,d11,d14,d104,d38,d63,d1625, d_base[0] ])
    Information_disclosure = concatenate([ d2491,d1625,d115,d12,d14,d104,d63,d131,d11,d38,d28,d12_2,d20,d36,d55,d114, d_base[1] ])
    Support = concatenate([ d337,d356,d25,d768,d55,d114,d20,d49,d12_2,d17,d27,d304,d28,d36,d67,d190,d21,d88,d156,d42,d11,d14,d12,d115,d104,d63,d38,d131, d_base[2] ])
    General_support = concatenate([ d42,d67,d36,d11,d20,d12_2,d49,d17,d27,d304,d28,d38, d_base[3] ])
    Info_support = concatenate([ d115,d12,d104,d14,d21,d88,d11,d12_2,d17,d27,d25,d768, d_base[4] ])
    Emo_support = concatenate([ d67,d190,d21,d12,d14,d11,d36,d114,d55,d20,d12_2,d17,d49,d25,d356, d_base[5]  ])

    stack = [Emotional_disclosure,Information_disclosure,Support,General_support,Info_support,Emo_support]

    # stack = [SeqSelfAttention(attention_activation=hyper_params['activation'])(i) for i in stack]
    stack = [MultiHeadAttention(head_num=1)(lyer) for i,lyer in enumerate(stack) ]
    # stack = [Dense(512, activation=hyper_params['activation'])(lyer) for i,lyer in enumerate(stack)]

    # stack = [Conv1D(filters=1, kernel_size=10, strides=1, padding='valid', activation=hyper_params['activation'])(i) for i in stack]
    stack = [Flatten()(i) for i in stack]

    output = [Dense(1, activation='sigmoid', name=labels[i])(lyer) for i,lyer in enumerate(stack)]

    model = keras.Model(inputs=input_text, outputs=output) #? Loss shift to a specific task during training?
    return model




def keras_bert_group_base(hyper_params):
    input_text = keras.Input(shape=hyper_params['input_shape'], dtype='float', name='trainable_input')
    base = 16
    flat = Dense(2048*base, activation=hyper_params['activation'])(input_text)
    flat = input_text
    # stack = [Dense(128, activation=hyper_params['activation'])(MultiHeadAttention(head_num=4)(input_text)) for i in range(6)]
    # dense = [Dense(label_set[labels[i]], activation=hyper_params['activation'], name='dense_'+labels[i])(flat) for i in range(hyper_params['T'])]  #? what is remove this layer completedly

    stack = [Dense(512*base, activation=hyper_params['activation'])(flat) for i in range(hyper_params['T'])]

    stack = [Dense(128, activation=hyper_params['activation'])(lyer) for i,lyer in enumerate(stack)]
    output = [Dense(1, activation='sigmoid', name=labels[i])(lyer) for i,lyer in enumerate(stack)]

    model = keras.Model(inputs=input_text, outputs=output) #? Loss shift to a specific task during training?
    return model

def keras_bert_group_base2(hyper_params):
    input_text = keras.Input(shape=hyper_params['input_shape'], dtype='float', name='trainable_input')
    base = 1
    # flat = Bidirectional(LSTM(256, return_sequences=True))(input_text)
    flat = Dense(int(1024*base), activation=hyper_params['activation'])(input_text)
    # stack = [Dense(128, activation=hyper_params['activation'])(MultiHeadAttention(head_num=4)(input_text)) for i in range(6)]
    # dense = [Dense(label_set[labels[i]], activation=hyper_params['activation'], name='dense_'+labels[i])(flat) for i in range(hyper_params['T'])]  #? what is remove this layer completedly

    stack = [Dense(int(1443*base), activation=hyper_params['activation'])(flat) for i in range(hyper_params['T'])]

    # stack = [SeqSelfAttention(attention_activation=hyper_params['activation'])(i) for i in stack]
    # stack = [MultiHeadAttention(head_num=1)(lyer) for i,lyer in enumerate(stack) ]
    # stack = [Dense(512, activation=hyper_params['activation'])(lyer) for i,lyer in enumerate(stack)]

    # stack = [Conv1D(filters=1, kernel_size=10, strides=1, padding='valid', activation=hyper_params['activation'])(i) for i in stack]
    stack = [Flatten()(i) for i in stack]

    output = [Dense(1, activation='sigmoid', name=labels[i])(lyer) for i,lyer in enumerate(stack)]

    model = keras.Model(inputs=input_text, outputs=output) #? Loss shift to a specific task during training?
    return model




def keras_bert_group_attention(hyper_params):
    '''
    Group1: Information_disclosure, Info_support
    Group2: Emotional_disclosure, Emo_support
    Group3: Support, General_support
    labels = ['Emotional_disclosure','Information_disclosure','Support','General_support','Info_support','Emo_support']
    '''
    assert(len(labels) == 6)
    input_text = keras.Input(shape=hyper_params['input_shape'], dtype='float', name='trainable_input')
    # conv = keras.layers.Conv1D(filters=512, kernel_size=10, strides=1, padding='same', activation=hyper_params['activation'])(input_text)
    # conv = Dropout(hyper_params['drop_rate'])(conv)
    # hidden1 = Bidirectional(LSTM(256, return_sequences=False))(conv)
    # hidden1 = BatchNormalization(axis=-1)(hidden1)
    hidden1 = input_text



    #distribution layers
    # d = [Dense(distribution_nodes, activation=hyper_params['activation'])(hidden1) for i in range(hyper_params['T'])]
    base = 128
    double_nodes = base*2
    triple_nodes = base*3
    head_num = 2
    nodes = 512
    # MultiHeadAttention(head_num=head_num)
    # Dense(double_nodes, activation=hyper_params['activation'])(MultiHeadAttention(head_num=head_num)(hidden1))
    # hidden1 = Dense(nodes, activation=hyper_params['activation'])(MultiHeadAttention(head_num=head_num)(hidden1))

    infod_s_infos = Dense(nodes, activation=hyper_params['activation'])(MultiHeadAttention(head_num=head_num)(hidden1))
    s_gs_emos = Dense(nodes, activation=hyper_params['activation'])(MultiHeadAttention(head_num=head_num)(hidden1))


    infod_s = Dense(nodes, activation=hyper_params['activation'])(MultiHeadAttention(head_num=head_num)(infod_s_infos))
    infod_infos = Dense(nodes, activation=hyper_params['activation'])(MultiHeadAttention(head_num=head_num)(infod_s_infos))
    s_infos = Dense(nodes, activation=hyper_params['activation'])(MultiHeadAttention(head_num=head_num)(infod_s_infos))

    s_gs = Dense(nodes, activation=hyper_params['activation'])(MultiHeadAttention(head_num=head_num)(s_gs_emos))
    s_emos = Dense(nodes, activation=hyper_params['activation'])(MultiHeadAttention(head_num=head_num)(s_gs_emos))
    gs_emos = Dense(nodes, activation=hyper_params['activation'])(MultiHeadAttention(head_num=head_num)(s_gs_emos))

    infod_emod = Dense(nodes, activation=hyper_params['activation'])(MultiHeadAttention(head_num=head_num)(hidden1))
    emod_emos = Dense(nodes, activation=hyper_params['activation'])(MultiHeadAttention(head_num=head_num)(hidden1))


    #task_layers
    task_nodes = 256
    head_num2 = 1
    infod = Dense(task_nodes, activation=hyper_params['activation'])(MultiHeadAttention(head_num=3*head_num2)(concatenate([infod_s, infod_infos, hidden1])))
    emod = Dense(task_nodes, activation=hyper_params['activation'])(MultiHeadAttention(head_num=3*head_num2)(concatenate([infod_emod, emod_emos, hidden1])))
    s = Dense(task_nodes, activation=hyper_params['activation'])(MultiHeadAttention(head_num=5*head_num2)(concatenate([s_infos, s_gs, s_emos, gs_emos, hidden1])))
    gs = Dense(task_nodes, activation=hyper_params['activation'])(MultiHeadAttention(head_num=3*head_num2)(concatenate([s_gs, gs_emos, hidden1])))
    infos = Dense(task_nodes, activation=hyper_params['activation'])(MultiHeadAttention(head_num=3*head_num2)(concatenate([infod_infos, s_infos, hidden1])))
    emos = Dense(task_nodes, activation=hyper_params['activation'])(MultiHeadAttention(head_num=4*head_num2)(concatenate([s_emos, gs_emos, emod_emos, hidden1])))

    stack = [infod, emod, s, gs, infos, emos]
    # mid = [Dense(512, activation=hyper_params['activation'])(lyer) for i,lyer in enumerate(stack)]
    stack = [Softmax()(i) for i in stack]
    stack = [Conv1D(filters=1, kernel_size=10, strides=1, padding='same', activation=hyper_params['activation'])(i) for i in stack]
    stack = [Flatten()(i) for i in stack]



    # output = [Dense(1, activation='sigmoid', name=labels[i])(lyer) for i,lyer in enumerate(stack)]
    output = [Dense(1, activation=hyper_params['activation'])(MultiHeadAttention(head_num=head_num)(hidden1)) for i in range(6)]

    model = keras.Model(inputs=input_text, outputs=output) #? Loss shift to a specific task during training?
    return model


def keras_bert_group_sa(hyper_params):
    assert(len(labels) == 6)
    input_text = keras.Input(shape=hyper_params['input_shape'], dtype='float', name='trainable_input')
    # hidden1 = input_text
    # conv = keras.layers.Conv1D(filters=512, kernel_size=10, strides=1, padding='same', activation=hyper_params['activation'])(input_text)
    # hidden1 = Flatten()(input_text)
    hidden1 = input_text
    # tmp = keras.layers.Bidirectional(keras.layers.LSTM(units=128, return_sequences=True))(input_text)
    # hidden1 = SeqSelfAttention(attention_activation='sigmoid')(tmp)

    #distribution layers
    distribution_nodes = 128
    return_sequences = True
    # d = [Dense(distribution_nodes, activation=hyper_params['activation'])(hidden1) for i in range(hyper_params['T'])]
    # distribution_layer = Dense(distribution_nodes, activation=hyper_params['activation'])
    # distribution_layer = Bidirectional(LSTM(units=128, return_sequences=True))
    infod = Bidirectional(LSTM(units=distribution_nodes, return_sequences=return_sequences))(hidden1)
    emod = Bidirectional(LSTM(units=distribution_nodes, return_sequences=return_sequences))(hidden1)
    s = Bidirectional(LSTM(units=distribution_nodes, return_sequences=return_sequences))(hidden1)
    gs = Bidirectional(LSTM(units=distribution_nodes, return_sequences=return_sequences))(hidden1)
    infos = Bidirectional(LSTM(units=distribution_nodes, return_sequences=return_sequences))(hidden1)
    emos = Bidirectional(LSTM(units=distribution_nodes, return_sequences=return_sequences))(hidden1)

    #chi2 layer paris
    head_num = 2
    # att_layer = MultiHeadAttention(head_num=head_num, name='Multi-Head')(input_layer)
    # SeqSelfAttention(attention_activation='sigmoid')
    # ssattention = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL, attention_activation=None, kernel_regularizer=keras.regularizers.l2(1e-6), use_attention_bias=False)
    infodemod = SeqSelfAttention(attention_activation='sigmoid')(concatenate([infod, emod]))
    infods = SeqSelfAttention(attention_activation='sigmoid')(concatenate([infod, s]))
    infodinfos = SeqSelfAttention(attention_activation='sigmoid')(concatenate([infod, infos]))
    emodemos = SeqSelfAttention(attention_activation='sigmoid')(concatenate([emod, emos]))
    sgs = SeqSelfAttention(attention_activation='sigmoid')(concatenate([s, gs]))
    sinfos = SeqSelfAttention(attention_activation='sigmoid')(concatenate([s, infos]))
    semos = SeqSelfAttention(attention_activation='sigmoid')(concatenate([s, emos]))
    gsemos = SeqSelfAttention(attention_activation='sigmoid')(concatenate([gs, emos]))

    #task_layers
    task_nodes = 128
    head_num2 = 4
    tinfod = Dense(task_nodes, activation=hyper_params['activation'])(SeqSelfAttention(attention_activation='sigmoid')(concatenate([infod, infodemod, infods, infodinfos])))
    temod = Dense(task_nodes, activation=hyper_params['activation'])(SeqSelfAttention(attention_activation='sigmoid')(concatenate([emod, infodemod, emodemos])))
    ts = Dense(task_nodes, activation=hyper_params['activation'])(SeqSelfAttention(attention_activation='sigmoid')(concatenate([s, infods, sgs, sinfos, semos])))
    tgs = Dense(task_nodes, activation=hyper_params['activation'])(SeqSelfAttention(attention_activation='sigmoid')(concatenate([gs, sgs, gsemos])))
    tinfos = Dense(task_nodes, activation=hyper_params['activation'])(SeqSelfAttention(attention_activation='sigmoid')(concatenate([infos, infodinfos, sinfos])))
    temos = Dense(task_nodes, activation=hyper_params['activation'])(SeqSelfAttention(attention_activation='sigmoid')(concatenate([emos, emodemos, semos, gsemos])))

    stack = [tinfod, temod, ts, tgs, tinfos, temos]
    mid = [Dense(128, activation=hyper_params['activation'])(lyer) for i,lyer in enumerate(stack)]
    mid2 = [Flatten()(i) for i in mid]

    output = [Dense(1, activation='sigmoid', name=labels[i])(lyer) for i,lyer in enumerate(mid2)]
    model = keras.Model(inputs=input_text, outputs=output) #? Loss shift to a specific task during training?
    return model


def keras_bert_group_mh(hyper_params):
    assert(len(labels) == 6)
    input_text = keras.Input(shape=hyper_params['input_shape'], dtype='float', name='trainable_input')
    # hidden1 = input_text
    # conv = keras.layers.Conv1D(filters=512, kernel_size=10, strides=1, padding='same', activation=hyper_params['activation'])(input_text)
    # hidden1 = Flatten()(input_text)
    hidden1 = input_text
    # tmp = keras.layers.Bidirectional(keras.layers.LSTM(units=128, return_sequences=True))(input_text)
    # hidden1 = SeqSelfAttention(attention_activation='sigmoid')(tmp)

    #distribution layers
    # d = [Dense(128, activation=hyper_params['activation'])(hidden1) for i in range(hyper_params['T'])]
    # distribution_layer = Bidirectional(LSTM(units=128, return_sequences=True))
    infod = Bidirectional(LSTM(units=128, return_sequences=True))(hidden1)
    emod = Bidirectional(LSTM(units=128, return_sequences=True))(hidden1)
    s = Bidirectional(LSTM(units=128, return_sequences=True))(hidden1)
    gs = Bidirectional(LSTM(units=128, return_sequences=True))(hidden1)
    infos = Bidirectional(LSTM(units=128, return_sequences=True))(hidden1)
    emos = Bidirectional(LSTM(units=128, return_sequences=True))(hidden1)

    #chi2 layer paris
    head_num = 2
    # att_layer = MultiHeadAttention(head_num=head_num, name='Multi-Head')(input_layer)
    # ssattention = MultiHeadAttention(head_num=head_num)
    infodemod = MultiHeadAttention(head_num=head_num)(concatenate([infod, emod]))
    infods = MultiHeadAttention(head_num=head_num)(concatenate([infod, s]))
    infodinfos = MultiHeadAttention(head_num=head_num)(concatenate([infod, infos]))
    emodemos = MultiHeadAttention(head_num=head_num)(concatenate([emod, emos]))
    sgs = MultiHeadAttention(head_num=head_num)(concatenate([s, gs]))
    sinfos = MultiHeadAttention(head_num=head_num)(concatenate([s, infos]))
    semos = MultiHeadAttention(head_num=head_num)(concatenate([s, emos]))
    gsemos = MultiHeadAttention(head_num=head_num)(concatenate([gs, emos]))

    #task_layers
    task_nodes = 128
    # head_num2 = 4
    tinfod = Dense(task_nodes, activation=hyper_params['activation'])(MultiHeadAttention(head_num=7, name='tinfod')(concatenate([infod, infodemod, infods, infodinfos])))
    temod = Dense(task_nodes, activation=hyper_params['activation'])(MultiHeadAttention(head_num=5, name='temod')(concatenate([emod, infodemod, emodemos])))
    ts = Dense(task_nodes, activation=hyper_params['activation'])(MultiHeadAttention(head_num=9, name='ts')(concatenate([s, infods, sgs, sinfos, semos])))
    tgs = Dense(task_nodes, activation=hyper_params['activation'])(MultiHeadAttention(head_num=5, name='tgs')(concatenate([gs, sgs, gsemos])))
    tinfos = Dense(task_nodes, activation=hyper_params['activation'])(MultiHeadAttention(head_num=5, name='tinfos')(concatenate([infos, infodinfos, sinfos])))
    temos = Dense(task_nodes, activation=hyper_params['activation'])(MultiHeadAttention(head_num=7, name='temos')(concatenate([emos, emodemos, semos, gsemos])))
    # tinfod = MultiHeadAttention(head_num=7, name='tinfod')(concatenate([infod, infodemod, infods, infodinfos]))
    # temod = MultiHeadAttention(head_num=5, name='temod')(concatenate([emod, infodemod, emodemos]))
    # ts = MultiHeadAttention(head_num=9, name='ts')(concatenate([s, infods, sgs, sinfos, semos]))
    # tgs = MultiHeadAttention(head_num=5, name='tgs')(concatenate([gs, sgs, gsemos]))
    # tinfos = MultiHeadAttention(head_num=5, name='tinfos')(concatenate([infos, infodinfos, sinfos]))
    # temos = MultiHeadAttention(head_num=7, name='temos')(concatenate([emos, emodemos, semos, gsemos]))


    stack = [tinfod, temod, ts, tgs, tinfos, temos]
    # mid = [Dense(128, activation=hyper_params['activation'])(lyer) for i,lyer in enumerate(stack)]
    # mid2 = [Flatten()(i) for i in mid]

    mid = [LSTM(units=64, return_sequences=False)(lyer) for i,lyer in enumerate(stack)]
    

    output = [Dense(1, activation='sigmoid', name=labels[i])(lyer) for i,lyer in enumerate(mid)]
    model = keras.Model(inputs=input_text, outputs=output) #? Loss shift to a specific task during training?
    return model


def keras_bert_group_mh2(hyper_params):
    assert(len(labels) == 6)
    input_text = keras.Input(shape=hyper_params['input_shape'], dtype='float', name='trainable_input')
    # hidden1 = input_text
    conv = keras.layers.Conv1D(filters=512, kernel_size=10, strides=1, padding='same', activation=hyper_params['activation'])(input_text)
    hidden1 = Bidirectional(LSTM(256))(conv)
    # hidden1 = Flatten()(input_text)

    double_nodes = 256
    triple_nodes = 384
    infod_s_infos = Dense(triple_nodes, activation=hyper_params['activation'])(hidden1)
    s_gs_emos = Dense(triple_nodes, activation=hyper_params['activation'])(hidden1)

    infod_emod = Dense(double_nodes, activation=hyper_params['activation'])(hidden1)
    emod_emos = Dense(double_nodes, activation=hyper_params['activation'])(hidden1)

    #task_layers
    task_nodes = 128
    # head_num2 = 4
    # infod = Dense(task_nodes, activation=hyper_params['activation'])(MultiHeadAttention(head_num=5, name='infod')(concatenate([infod_s_infos, infod_emod])))
    # emod = Dense(task_nodes, activation=hyper_params['activation'])(MultiHeadAttention(head_num=4, name='emod')(concatenate([infod_emod, emod_emos])))
    # s = Dense(task_nodes, activation=hyper_params['activation'])(MultiHeadAttention(head_num=6, name='s')(concatenate([infod_s_infos, s_gs_emos])))
    # gs = Dense(task_nodes, activation=hyper_params['activation'])(s_gs_emos)
    # infos = Dense(task_nodes, activation=hyper_params['activation'])(infod_s_infos)
    # emos = Dense(task_nodes, activation=hyper_params['activation'])(MultiHeadAttention(head_num=5, name='emos')(concatenate([s_gs_emos, emod_emos])))
    infod = Dense(task_nodes, activation=hyper_params['activation'])(concatenate([infod_s_infos, infod_emod]))
    emod = Dense(task_nodes, activation=hyper_params['activation'])(concatenate([infod_emod, emod_emos]))
    s = Dense(task_nodes, activation=hyper_params['activation'])(concatenate([infod_s_infos, s_gs_emos]))
    gs = Dense(task_nodes, activation=hyper_params['activation'])(s_gs_emos)
    infos = Dense(task_nodes, activation=hyper_params['activation'])(infod_s_infos)
    emos = Dense(task_nodes, activation=hyper_params['activation'])(concatenate([s_gs_emos, emod_emos]))


    stack = [infod, emod, s, gs, infos, emos]
    mid = [Dense(64, activation=hyper_params['activation'])(lyer) for i,lyer in enumerate(stack)]
    # mid2 = [Flatten()(i) for i in mid]

    # mid = [LSTM(units=64, return_sequences=False)(lyer) for i,lyer in enumerate(stack)]
    output = [Dense(1, activation='sigmoid', name=labels[i])(lyer) for i,lyer in enumerate(mid)]
    model = keras.Model(inputs=input_text, outputs=output) #? Loss shift to a specific task during training?
    return model


def keras_bert_group_mh3(hyper_params):
    assert(len(labels) == 6)
    input_text = keras.Input(shape=hyper_params['input_shape'], dtype='float', name='trainable_input')
    # hidden1 = input_text
    # conv = keras.layers.Conv1D(filters=512, kernel_size=10, strides=1, padding='same', activation=hyper_params['activation'])(input_text)
    hidden1 = Bidirectional(LSTM(256, return_sequences=True))(input_text)
    # hidden1 = Flatten()(input_text)

    double_nodes = 256*2
    triple_nodes = 256*3
    infod_s_infos = Dense(triple_nodes, activation=hyper_params['activation'])(hidden1)
    s_gs_emos = Dense(triple_nodes, activation=hyper_params['activation'])(hidden1)

    infod_emod = Dense(double_nodes, activation=hyper_params['activation'])(hidden1)
    emod_emos = Dense(double_nodes, activation=hyper_params['activation'])(hidden1)

    #task_layers
    task_nodes = 256
    base = 4
    bias = 2
    # head_num2 = 4
    infod = Dense(task_nodes, activation=hyper_params['activation'])(MultiHeadAttention(head_num=(5+bias)*base, name='infod')(concatenate([infod_s_infos, infod_emod, hidden1])))
    emod = Dense(task_nodes, activation=hyper_params['activation'])(MultiHeadAttention(head_num=(4+bias)*base, name='emod')(concatenate([infod_emod, emod_emos, hidden1])))
    s = Dense(task_nodes, activation=hyper_params['activation'])(MultiHeadAttention(head_num=(6+bias)*base, name='s')(concatenate([infod_s_infos, s_gs_emos, hidden1])))
    gs = Dense(task_nodes, activation=hyper_params['activation'])(MultiHeadAttention(head_num=(3+bias)*base, name='gs')(concatenate([s_gs_emos, hidden1])))
    infos = Dense(task_nodes, activation=hyper_params['activation'])(MultiHeadAttention(head_num=(3+bias)*base, name='infos')(concatenate([infod_s_infos, hidden1])))
    emos = Dense(task_nodes, activation=hyper_params['activation'])(MultiHeadAttention(head_num=(5+bias)*base, name='emos')(concatenate([s_gs_emos, emod_emos, hidden1])))


    stack = [infod, emod, s, gs, infos, emos]
    mid = [Dense(64, activation=hyper_params['activation'])(lyer) for i,lyer in enumerate(stack)]
    # mid2 = [Flatten()(i) for i in mid]

    mid = [LSTM(units=64, return_sequences=False)(lyer) for i,lyer in enumerate(stack)]
    output = [Dense(1, activation='sigmoid', name=labels[i])(lyer) for i,lyer in enumerate(mid)]
    model = keras.Model(inputs=input_text, outputs=output) #? Loss shift to a specific task during training?
    return model




def keras_bert_liwc_asinput(hyper_params):
    input_text = keras.Input(shape=hyper_params['input_shape'], dtype='float', name='trainable_input')

    if len(hyper_params['input_shape']) == 2:
        conv = keras.layers.Conv1D(filters=512, kernel_size=10, strides=1, padding='same', activation=hyper_params['activation'])(input_text)

        avg_pool = AveragePooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(conv)
        max_pool = MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(conv)
        x = concatenate([avg_pool, max_pool])
        flat = Flatten(name='flat')(x)
    elif len(hyper_params['input_shape']) == 1:
        flat = Dense(512, activation=hyper_params['activation'], name='Dense')(input_text)

    hidden1 = Dense(512, activation=hyper_params['activation'], name='hidden1')(flat)

    input_liwc = keras.Input(shape=(92,), dtype='float', name='liwc_input')
    # hidden2 = Dense(128, activation='sigmoid', name='hidden2')(input_liwc)

    hidden = concatenate([hidden1, input_liwc])

    # dense = [CustomizedDenseLayer(output_dim=32)(conc) for _ in range(hyper_params['T'])]  #? what is remove this layer completedly
    dense = [Dense(64, activation=hyper_params['activation'], name='dense_'+labels[i])(hidden) for i in range(hyper_params['T'])]  #? what is remove this layer completedly
    # dense = ''

    output = [ Dense(1, activation='sigmoid', name=labels[i])(lyer) for i,lyer in enumerate(dense)]
    # output = [ Dense(1, activation='sigmoid' if hyper_params['isDataBinary'] else hyper_params['activation'])(flat) for _ in range(hyper_params['T'])]

    model = keras.Model(inputs=[input_text,input_liwc], outputs=output) #? Loss shift to a specific task during training?
    # print the model summary
    print(model.summary())
    return model

def keras_bert_liwc_asoutput(hyper_params):
    input_text = keras.Input(shape=hyper_params['input_shape'], dtype='float', name='trainable_input')

    if len(hyper_params['input_shape']) == 2:
        conv = keras.layers.Conv1D(filters=512, kernel_size=10, strides=1, padding='same', activation=hyper_params['activation'])(input_text)

        avg_pool = AveragePooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(conv)
        max_pool = MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(conv)
        x = concatenate([avg_pool, max_pool])
        flat = Flatten(name='flat')(x)
    elif len(hyper_params['input_shape']) == 1:
        flat = Dense(512, activation=hyper_params['activation'], name='Dense')(input_text)

    hidden = Dense(512, activation=hyper_params['activation'], name='hidden')(flat)

    # hidden2 = Dense(128, activation='sigmoid', name='hidden2')(input_liwc)

    # dense = [CustomizedDenseLayer(output_dim=32)(conc) for _ in range(hyper_params['T'])]  #? what is remove this layer completedly
    dense = [Dense(128, activation=hyper_params['activation'], name='dense_'+labels[i])(hidden) for i in range(hyper_params['T'])]  #? what is remove this layer completedly
    # dense = ''

    output_liwc = Dense(92, activation='sigmoid', name='liwc')(hidden)
    output = [ Dense(1, activation='sigmoid', name=labels[i])(lyer) for i,lyer in enumerate(dense)]
    output.append(output_liwc)
    # output = [ Dense(1, activation='sigmoid' if hyper_params['isDataBinary'] else hyper_params['activation'])(flat) for _ in range(hyper_params['T'])]
    

    model = keras.Model(inputs=input_text, outputs=output) #? Loss shift to a specific task during training?
    # print the model summary
    print(model.summary())
    return model



def keras_bert_leakyrelu(hyper_params):
    input_text = keras.Input(shape=hyper_params['input_shape'], dtype='float', name='trainable_input')

    if len(hyper_params['input_shape']) == 2:
        conv = keras.layers.Conv1D(filters=512, kernel_size=10, strides=1, padding='same', activation=hyper_params['activation'])(input_text)
        # conv = Bidirectional(GRU(512, return_sequences=True))(input_text)

        avg_pool = AveragePooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(conv)
        max_pool = MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(conv)
        x = concatenate([avg_pool, max_pool])

        flat = Flatten(name='flat')(x)
    elif len(hyper_params['input_shape']) == 1:
        flat = Dense(512, name='Dense', activation=hyper_params['activation'])(input_text)

    hidden1 = Dense(512, name='hidden1', activation=hyper_params['activation'])(flat)

    # dense = [CustomizedDenseLayer(output_dim=32)(conc) for _ in range(hyper_params['T'])]  #? what is remove this layer completedly
    dense = [Dense(64, name='dense_'+labels[i], activation=hyper_params['activation'])(hidden1) for i in range(hyper_params['T'])]  #? what is remove this layer completedly
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



def keras_bert_triple(hyper_params):
    conv_filters = 512
    conv_kernel = 10
    conv_strides = 1

    bert_input = keras.Input(shape=(34, 1024), dtype='float', name='trainable_bert_input')

    elmo_input = keras.Input(shape=(34, 1024), dtype='float', name='trainable_elmo_input')

    glove_input = keras.Input(shape=(34, 200), dtype='float', name='trainable_glove_input')

    conv = [keras.layers.Conv1D(filters=conv_filters, kernel_size=conv_kernel, strides=conv_strides, padding='same', activation=hyper_params['activation'])(i)  for i in (bert_input, elmo_input, glove_input)]
    avg_pool = [AveragePooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(i) for i in conv]
    max_pool = [MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(i) for i in conv]
    conc1 = [concatenate([i,j]) for i,j in zip(avg_pool, max_pool)]
    flat_1 = [Flatten()(i) for i in conc1]

    blstm = [Bidirectional(LSTM(256))(i) for i in (bert_input, elmo_input, glove_input)]
    # avg_pool_2 = [AveragePooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(i) for i in blstm]
    # max_pool_2 = [MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(i) for i in blstm]
    # conc2 = [concatenate([i,j]) for i,j in zip(avg_pool_2, max_pool_2)]
    flat_2 = blstm
    
    bgru = [Bidirectional(GRU(128, return_sequences=True))(i)  for i in (bert_input, elmo_input, glove_input)]
    avg_pool_3 = [AveragePooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(i) for i in bgru]
    max_pool_3 = [MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(i) for i in bgru]
    conc3 = [concatenate([i,j]) for i,j in zip(avg_pool_3, max_pool_3)]
    flat_3 = [Flatten()(i) for i in conc3]
    

    conc123 = [concatenate(i) for i in (flat_1, flat_2, flat_3)]
    conc = concatenate(conc123)

    hidden1 = Dense(512, activation=hyper_params['activation'], name='hidden1')(conc)

    # dense = [CustomizedDenseLayer(output_dim=32)(conc) for _ in range(hyper_params['T'])]  #? what is remove this layer completedly
    dense = [Dense(64, activation=hyper_params['activation'], name='dense_'+labels[i])(hidden1) for i in range(hyper_params['T'])]  #? what is remove this layer completedly
    # dense = ''

    output = [ Dense(1, activation='sigmoid', name=labels[i])(lyer) for i,lyer in enumerate(dense)]
    # output = [ Dense(1, activation='sigmoid' if hyper_params['isDataBinary'] else hyper_params['activation'])(flat) for _ in range(hyper_params['T'])]

    model = keras.Model(inputs=[bert_input, elmo_input, glove_input], outputs=output) #? Loss shift to a specific task during training?
    # print the model summary
    print(model.summary())
    return model
    