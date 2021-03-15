'''
main.py
'''
hyper_params = {
    'isDataBinary': True,
    'balance_label': False,
    'learning_rate': 1e-3,
    'n_splits': 1,
    'maxlen': 32,
    'epochs': 10,
    'batch_size': 512,
    'embedding_dim': 200,
    'max_words': -1,    # Adjust later
    'weights': {},      # Adjust later if using GloVe
    'T': len(labels),
    'optimizer': 'adam',
    'train_able': True,
    'loss': 'binary_crossentropy',
    'activation': 'relu',
    'dropout_rate': 0.5,
    'elmo_tag': 'default', # 'default' [None,1024] or 'elmo' [None, 50, 1024] or 'word_emb' [None, 50, 512]
    'embedding_trainable': True, # No use if elmo_tag==word_emb
    'elmo_signature': 'signature',  # 'signature' if input is (1,) strings or 'tokens' if input is (50,) tokens
}
model, X_test, y_test, y_pred, history = build_model(X, Y, model_name = 'keras_elmo', hyper_params=hyper_params, metrics=['accuracy',precision_m, recall_m, f1_m,])


'''
build_model.py
'''
def balance_label(X, Y):
    Y = np.array(Y)
    transpose = (Y.shape[0] == len(labels))
    print(Y)
    print('transpose: {}'.format(transpose))
    if transpose:
        Y = Y.transpose()
    print('Balance label:')
    rates = [0 for _ in range(len(labels))]
    for i in range(len(labels)):
        num = sum( 1 if i>0 else 0 for i in Y[:,i] )
        rates[i] = math.ceil(len(X)*1.0/num)

    print('extend rates:')
    print(rates)
    time.sleep(10)
    X2 = []
    Y2 = []
    for x,y in zip(X, Y):
        rate = max( rates[i] if y[i]>0 else 1 for i in range(len(labels)) )
        for _ in range(rate):
            X2.append(x)
            Y2.append(y)
    X2, Y2 = shuffle(X2, Y2)

    print('## Balanced data:')
    print('length: {}'.format(len(Y2)))
    # print(Y2)
    showinfo(Y2)

    if transpose:
        Y2 = np.array(Y2).transpose()
    
    return np.array(X2),np.array(Y2)



'''
build_model.py
'''
def build_model_elmo(X, Y, model_name, hyper_params, metrics=METRICS):
    X,t = token_prepare(X, as_index=('glove' in model_name), as_string=('elmo' in model_name or 'bert' in model_name))

    hyper_params['max_words'] = unique_token_num + 2 

    if 'glove' in model_name:
        from model_glove import get_embedding_matrix
        hyper_params['weights'] = get_embedding_matrix(hyper_params['embedding_dim'], t)
    
    print('hyper_params: {}'.format(hyper_params))

    f1,acc,auc,preci,recall = [],[],[],[],[]

    # kf = StratifiedKFold(n_splits=hyper_params['n_splits'] if hyper_params['n_splits']!=1 else 5, random_state=7001, shuffle=True)
    sss = StratifiedShuffleSplit(n_splits=hyper_params['n_splits'] if hyper_params['n_splits']!=1 else 5, random_state=7001)

    for train_index, test_index in sss.split(X, Y if Y.shape[-1]==1 else Y[:,1] ):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        # if hyper_params['balance_label']:
        #     X_train, y_train = balance_label(X_train, y_train)

        # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

        model = eval(model_name)(hyper_params)
        model.compile(optimizer=Adam(lr=hyper_params['learning_rate']), loss=hyper_params['loss'], metrics=metrics) #['acc',f1_m,precision_m, recall_m]
        history = model.fit(x=X_train, y=[ i.squeeze() for i in y_train.T], 
                    epochs=hyper_params['epochs'], 
                    batch_size=hyper_params['batch_size'],
                    validation_split=0.25)

        print('Model {} finished.'.format(model_name))

        from ploting import plot_metrics
        plot_metrics(history)

        # return model, X_test, y_test
        y_pred = model.predict(X_test)
        r1,r2,r3,r4,r5 = test_model(y_test, y_pred)
        f1.append(r1);acc.append(r2);auc.append(r3);preci.append(r4);recall.append(r5)

        if hyper_params['n_splits']==1:
            break

    f = lambda x,y: np.mean(np.array(x)[:,y])
    print('\n\n##### Cross Validation result:  #######\n')
    for i in range(len(labels)):
        print('Label {}:'.format(labels[i]))
        print('Average f1_score: %.3f'%f(f1,i))
        print('Average accuracy_score: %.3f'%f(acc,i))
        print('Average auc_score: %.3f'%f(auc,i))
        print('Average precision_score: %.3f'%f(preci,i))
        print('Average recall_score: %.3f'%f(recall,i))
    return model, X_test, y_test, y_pred, history


'''
build_model.py
Same as build_model_bert, so delete
'''

def build_model_group(Y, hyper_params, metrics=METRICS):
    import pickle
    # bert_default_32.pickle, bert_no_pooling_32.pickle
    X = read_pickle(hyper_params['embedding_file'])


    f1,acc,auc,preci,recall = [],[],[],[],[]
    from sklearn.model_selection import train_test_split
    print('X shape: {}\n y shape: {}'.format(X.shape, Y.shape))

    X_train, X_test, y_train, y_test, train_indx, test_indx = train_test_split(X, Y, list(range(len(X))), test_size=0.33, random_state=42)
    print('X_train shape: {}\n y_train shape: {}\n X_test shape: {}\ny_test shape: {}'.format(X_train.shape, y_train.shape, X_test, y_test))

    model = keras_bert_group_venn2(hyper_params)
    # for i in model.layers:
    #     if 'trainable' in i.name:
    #         i.trainable = True
    model.compile(optimizer=Adam(lr=hyper_params['learning_rate']), loss=hyper_params['loss'], metrics=metrics) #['acc',f1_m,precision_m, recall_m]
    print(model.summary())
    print('y_train.shape: {}'.format(y_train.shape))
    history = model.fit(x=X_train, y=[ i.squeeze() for i in y_train.T], 
                    epochs=hyper_params['epochs'], 
                    batch_size=hyper_params['batch_size'],
                    class_weight=[dict(enumerate(class_weight.compute_class_weight('balanced', [0,1], i.squeeze()))) for i in y_train.T],
                    validation_data=(X_test, [ i.squeeze() for i in y_test.T]) # validation_split=0.25
                    )
    y_pred = model.predict(X_test)
    print('Totoal numebr of trainable variables:',np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
    r1,r2,r3,r4,r5 = test_model(y_test, y_pred, name=hyper_params['embedding_file']) #threshold=get_best_threshold(y_test, y_pred)
    f1.append(r1);acc.append(r2);auc.append(r3);preci.append(r4);recall.append(r5)
    return model, X_train, X_test, y_train, y_test, y_pred, train_indx, test_indx, history






'''
model_bert.py
'''
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

