#!/usr/local/bin/python3
# -*- coding: utf-8 -*- 

from header import *

from model_basic import *
from model_elmo import *
from model_glove import *
from model_bert import *






def text_process(sentences):
    # tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
    tokens = [text_to_word_sequence(sentence) for sentence in sentences]
    lengths = np.array([ len(i) for i in tokens ])
    max_len = lengths.max()
    min_len = lengths.min()
    mean_len = lengths.mean()
    print(max_len, mean_len)

    s = set()
    for i in tokens:
        s.update(i)
    unique_token_num = len(s)
    total_token_num = sum(lengths)

    cover_len = math.ceil(mean_len + 1.8 * lengths.std())

    print('\nMax token length: {}'.format(max_len))
    print('Min token length: {}'.format(min_len))
    print('Mean token length: %.3f'%(mean_len))
    print('Median token length: %.3f'%(np.median(lengths)))
    print('Number of unique tokens: {}'.format(unique_token_num))
    print('Total number of tokens: {}'.format(total_token_num))
    print('Percentage of sentences that over the mean token length: %.3f%%'%( sum(i >= mean_len for i in lengths)*100.0 / len(lengths) ))
    print('Modified cover length is: {}'.format(cover_len))
    print('Percentage of sentences that under the cover_len: %.3f%%\n\n'%( sum(i <= cover_len for i in lengths)*100.0 / len(lengths) ))
    return max_len,mean_len,cover_len,unique_token_num,total_token_num

def tokenizer(sentences, unique_token_num):
    t = tf.keras.preprocessing.text.Tokenizer(num_words=unique_token_num + 1, lower=False, oov_token='<UKN>')
    tokens = [text_to_word_sequence(sentence) for sentence in sentences]
    t.fit_on_texts(tokens)
    return t

def sequential_and_padding(t, sentences, cover_len):
    tokens = [text_to_word_sequence(sentence) for sentence in sentences]
    sequences = t.texts_to_sequences(tokens)
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=cover_len, padding='post')
    return sequences

def sequences_reformat(t, sequences, as_string=True, padding_char=''):
    d = dict( [i,word] for word, i in t.word_index.items() )
    emb = []
    for indx,s in enumerate(sequences):
        tmp = [ d[i] if i!=0 else padding_char for i in s] #? Is it safe using '.' as padding char
        if as_string:
            emb.append(' '.join(tmp).strip())
        else: # as list
            emb.append(tmp)
    emb = np.array(emb)
    return emb


max_len,mean_len,cover_len,unique_token_num,total_token_num = 0,0,0,0,0
def token_prepare(sentences, as_index=False, as_string=True, padding_char='.'):
    global max_len,mean_len,cover_len,unique_token_num,total_token_num
    max_len,mean_len,cover_len,unique_token_num,total_token_num = text_process(sentences)
    t = tokenizer(np.array(sentences), unique_token_num)
    sequences = sequential_and_padding(t, sentences, cover_len)
    if not as_index:
        sequences = sequences_reformat(t, sequences, as_string = as_string, padding_char = padding_char)
    return sequences,t


# @pysnooper.snoop()
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

def test_model(y_true, y_pred, name=None):
    # loss, acc = model.evaluate(X, y)
    # print("loss and accuracy on test data: loss = {}, accuracy = {}".format(loss, acc))
    # f1_score, accuracy_score, precision_score, recall_score, classification_report, roc_auc_score
    f1,auc,acc,preci,recall = [],[],[],[],[]
    binary_check = lambda x: round(x)
    binary_format = lambda y: np.array([ binary_check(i) for i in y.squeeze() ])

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    if y_true.shape[0]!=len(labels):
        y_true = y_true.T

    if y_pred.shape[0]!=len(labels):
        y_pred = y_pred.T

    for label, i,j in zip(labels, y_true, y_pred):
        print('\nLabel \"{}\":'.format(label))
        try:
            auc.append(roc_auc_score(i,j))
        except Exception as e:
            auc.append(0.0)
        
        print('auc_score: %.3f'%auc[-1])
        i,j = binary_format(i), binary_format(j)
        f1.append(f1_score(i, j))
        print('f1_score: %.3f'%f1[-1])
        acc.append(accuracy_score(i,j))
        print('accuracy_score: %.3f'%acc[-1])
        preci.append(precision_score(i,j))
        print('precision_score: %.3f'%preci[-1])
        recall.append(recall_score(i,j))
        print('recall_score: %.3f'%recall[-1])
    print('\n<------------------------------------------------>\n')

    add_avg = lambda x: list(x) + ['-----', np.mean(x)]
    markdown_output(xheader=['accuracy', 'precision', 'recall', 'f1'], yheader=list(labels) + ['-----', 'Average'], matrix=np.array([add_avg(acc),add_avg(preci),add_avg(recall),add_avg(f1)]).T, table_name=name)

    print('\n<------------------------------------------------>\n')

    print('Average f1_score: %.3f'%np.mean(f1))
    print('Average accuracy_score: %.3f'%np.mean(acc))
    print('Average precision_score: %.3f'%np.mean(preci))
    print('Average recall_score: %.3f'%np.mean(recall))
    return f1,acc,auc,preci,recall

# @pysnooper.snoop()
def build_model(X, Y, model_name, hyper_params, metrics=METRICS):
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

def build_model_bert(Y, hyper_params, metrics=METRICS):
    import pickle
    # bert_default_32.pickle, bert_no_pooling_32.pickle
    X = read_pickle(hyper_params['embedding_file'])


    f1,acc,auc,preci,recall = [],[],[],[],[]
    from sklearn.model_selection import train_test_split
    print('X shape: {}\n y shape: {}'.format(X.shape, Y.shape))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    print('X_train shape: {}\n y_train shape: {}\n X_test shape: {}\ny_test shape: {}'.format(X_train.shape, y_train.shape, X_test, y_test))

    model = keras_bert(hyper_params)
    for i in model.layers:
        if 'trainable' in i.name:
            i.trainable = True
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
    r1,r2,r3,r4,r5 = test_model(y_test, y_pred, name=hyper_params['embedding_file'])
    f1.append(r1);acc.append(r2);auc.append(r3);preci.append(r4);recall.append(r5)
    return model, X_train, X_test, y_train, y_test, y_pred, history



def build_model_bert_dual(Y, hyper_params, metrics=METRICS):
    import pickle
    # bert_default_32.pickle, bert_no_pooling_32.pickle
    pickle_in = open( os.path.join(root_folder, 'data/pickle/', hyper_params['embedding_file']),"rb")
    X = pickle.load(pickle_in)
    pickle_in.close()


    f1,acc,auc,preci,recall = [],[],[],[],[]
    from sklearn.model_selection import train_test_split
    print('X shape: {}\n y shape: {}'.format(X.shape, Y.shape))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    print('X_train shape: {}\n y_train shape: {}\n X_test shape: {}\ny_test shape: {}'.format(X_train.shape, y_train.shape, X_test, y_test))

    model = keras_bert_dual(hyper_params)
    for i in model.layers:
        if 'trainable' in i.name:
            i.trainable = True
    model.compile(optimizer=Adam(lr=hyper_params['learning_rate']), loss=hyper_params['loss'], metrics=metrics) #['acc',f1_m,precision_m, recall_m]
    print(model.summary())
    print('y_train.shape: {}'.format(y_train.shape))
    history = model.fit(x=[X_train, X_train], y=[ i.squeeze() for i in y_train.T], 
                    epochs=hyper_params['epochs'], 
                    batch_size=hyper_params['batch_size'],
                    class_weight=[dict(enumerate(class_weight.compute_class_weight('balanced', [0,1], i.squeeze()))) for i in y_train.T],
                    validation_data=([X_test, X_test], [ i.squeeze() for i in y_test.T]) # validation_split=0.25
                    )

    y_pred = model.predict([X_test, X_test])
    r1,r2,r3,r4,r5 = test_model(y_test, y_pred, name=hyper_params['embedding_file'])
    f1.append(r1);acc.append(r2);auc.append(r3);preci.append(r4);recall.append(r5)
    return model, X_train, X_test, y_train, y_test, y_pred, history


def build_model_triple(Y, hyper_params, metrics=METRICS):
    import pickle
    # bert_default_32.pickle, bert_no_pooling_32.pickle
    bertx = read_pickle('bert_no_pooling_34_12860x34x1024.pickle')
    elmox = read_pickle('elmo_34_12860x34x1024.pickle')
    glovex = read_pickle('glove_34_12860x34x200.pickle')


    f1,acc,auc,preci,recall = [],[],[],[],[]
    from sklearn.model_selection import train_test_split
    bertx_train, bertx_test, elmox_train, elmox_test, glovex_train, glovex_test, y_train, y_test = train_test_split(bertx, elmox, glovex, Y, test_size=0.33, random_state=42)

    model = keras_bert_triple(hyper_params)
    for i in model.layers:
        if 'trainable' in i.name:
            i.trainable = True
    model.compile(optimizer=Adam(lr=hyper_params['learning_rate']), loss=hyper_params['loss'], metrics=metrics) #['acc',f1_m,precision_m, recall_m]
    print(model.summary())
    print('y_train.shape: {}'.format(y_train.shape))
    history = model.fit(x=[bertx_train, elmox_train, glovex_train], y=[ i.squeeze() for i in y_train.T], 
                    epochs=hyper_params['epochs'], 
                    batch_size=hyper_params['batch_size'],
                    class_weight=[dict(enumerate(class_weight.compute_class_weight('balanced', [0,1], i.squeeze()))) for i in y_train.T],
                    validation_data=([bertx_test, elmox_test, glovex_test], [ i.squeeze() for i in y_test.T]) # validation_split=0.25
                    )
    y_pred = model.predict([bertx_test, elmox_test, glovex_test])
    r1,r2,r3,r4,r5 = test_model(y_test, y_pred, name=hyper_params['embedding_file'])
    f1.append(r1);acc.append(r2);auc.append(r3);preci.append(r4);recall.append(r5)
    return model, bertx_train, bertx_test, elmox_train, elmox_test, glovex_train, glovex_test, y_train, y_test, y_pred, history

