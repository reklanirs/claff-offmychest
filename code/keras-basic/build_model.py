#!/usr/local/bin/python3
# -*- coding: utf-8 -*- 

from header import *
from file_reader import *

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
    print('Percentage of sentences that under 34: %.3f%%\n\n'%( sum(i <= 34 for i in lengths)*100.0 / len(lengths) ))
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

def get_best_threshold(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    binary_check = lambda thshld,x: 1 if x>thshld else 0
    binary_format = lambda thshld,y: np.array([ binary_check(thshld,i) for i in y.squeeze() ])
    threshold = []
    if y_true.shape[0]!=len(labels):
        y_true = y_true.T
    if y_pred.shape[0]!=len(labels):
        y_pred = y_pred.T
    for indx,label, i,j in zip( list(range(len(labels))), labels, y_true, y_pred):
        r = 1.0
        l = 0.0
        eps = 0.001
        while r - l > eps:
            md = (l + r)/2.0
            mdd = (md + r)/2.0
            i1,j1 = binary_format(md, i), binary_format(md, j)
            i2,j2 = binary_format(mdd, i), binary_format(mdd, j)
            fmd,fmdd = f1_score(i1,j1),f1_score(i2,j2)
            # print('check {},{}  {},{}'.format(l,r,fmd,fmdd))
            if fmd > fmdd:
                r = mdd
            else:
                l = md
        threshold.append(l)
    return threshold

def test_model(y_true, y_pred, name=None, threshold=None, labels=labels):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if len(y_true.shape) == 1:
        y_true.resize((y_true.shape[0],1))
    if len(y_pred.shape) == 1:
        y_pred.resize((y_pred.shape[0],1))
    print('test_model\ny_true.shape:{}, y_pred.shape:{}'.format(y_true.shape, y_pred.shape))
    # loss, acc = model.evaluate(X, y)
    # print("loss and accuracy on test data: loss = {}, accuracy = {}".format(loss, acc))
    # f1_score, accuracy_score, precision_score, recall_score, classification_report, roc_auc_score
    f1,auc,acc,preci,recall = [],[],[],[],[]
    if threshold == None:
        threshold = [0.5 for _ in labels]
    binary_check = lambda thshld,x: 1 if x>thshld else 0
    binary_format = lambda thshld,y: np.array([ binary_check(thshld,i) for i in y.squeeze() ])

    if y_true.shape[0]!=len(labels):
        y_true = y_true.T

    if y_pred.shape[0]!=len(labels):
        y_pred = y_pred.T
    
    for indx,label, i,j in zip( list(range(len(labels))), labels, y_true, y_pred):
        # print('\nLabel \"{}\":'.format(label))
        try:
            auc.append(roc_auc_score(i,j))
        except Exception as e:
            auc.append(0.0)
        
        # print('auc_score: %.3f'%auc[-1])
        i,j = binary_format(threshold[indx],i), binary_format(threshold[indx],j)
        f1.append(f1_score(i, j))
        # print('f1_score: %.3f'%f1[-1])
        acc.append(accuracy_score(i,j))
        # print('accuracy_score: %.3f'%acc[-1])
        preci.append(precision_score(i,j))
        # print('precision_score: %.3f'%preci[-1])
        recall.append(recall_score(i,j))
        # print('recall_score: %.3f'%recall[-1])
    print('\n<------------------------------------------------>\n')

    add_avg = lambda x: list(x) + ['-----', np.mean(x)]
    markdown_output(xheader=['accuracy', 'precision', 'recall', 'f1'], yheader=list(labels) + ['-----', 'Average'], matrix=np.array([add_avg(acc),add_avg(preci),add_avg(recall),add_avg(f1)]).T, table_name=name)

    print('\n<------------------------------------------------>\n')

    print('Average f1_score: %.3f'%np.mean(f1))
    print('Average accuracy_score: %.3f'%np.mean(acc))
    print('Average precision_score: %.3f'%np.mean(preci))
    print('Average recall_score: %.3f'%np.mean(recall))
    return f1,acc,auc,preci,recall





def build_model_bert(Y, hyper_params, metrics=METRICS):
    import pickle
    # bert_default_32.pickle, bert_no_pooling_32.pickle
    X = read_pickle(hyper_params['embedding_file'])


    f1,acc,auc,preci,recall = [],[],[],[],[]
    from sklearn.model_selection import train_test_split
    print('X shape: {}\n y shape: {}'.format(X.shape, Y.shape))

    X_train, X_test, y_train, y_test, train_indx, test_indx = train_test_split(X, Y, list(range(len(X))), test_size=0.2, random_state=42)
    print('X_train shape: {}\n y_train shape: {}\n X_test shape: {}\ny_test shape: {}'.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

    model = eval(hyper_params['model_name'])(hyper_params)
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
    r1,r2,r3,r4,r5 = test_model(y_test, y_pred, name=hyper_params['embedding_file']) #threshold=get_best_threshold(y_test, y_pred)
    f1.append(r1);acc.append(r2);auc.append(r3);preci.append(r4);recall.append(r5)
    return model, X_train, X_test, y_train, y_test, y_pred, train_indx, test_indx, history





def build_model_kfold_train(X, Y, hyper_params, metrics, return_dict=[]):
    X_train, X_test, y_train, y_test, train_indx, test_indx = train_test_split(X, Y, list(range(len(X))), test_size=0.33, random_state=42)
    print('X_train shape: {}\n y_train shape: {}\n X_test shape: {}\ny_test shape: {}'.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

    from model_bert import keras_bert_group_venn,keras_bert_group_venn2
    model = eval(hyper_params['model_name'])(hyper_params)
    # for i in model.layers:
    #     if 'trainable' in i.name:
    #         i.trainable = True
    model._make_predict_function()
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
    return_dict.append((model, X_train, X_test, y_train, y_test, y_pred, train_indx, test_indx, history))
    return model, X_train, X_test, y_train, y_test, y_pred, train_indx, test_indx, history



def build_model_kfold_Process(Y, hyper_params, metrics=METRICS):
    import pickle
    # bert_default_32.pickle, bert_no_pooling_32.pickle
    X = read_pickle(hyper_params['embedding_file'])

    f1,acc,auc,preci,recall = [],[],[],[],[]
    from sklearn.model_selection import train_test_split
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    for _ in range(hyper_params['k_fold']):
        process_train = multiprocessing.Process(target=build_model_kfold_train, args=(X, Y, hyper_params, metrics, return_dict))
        process_train.start()
        process_train.join()
        
        model, X_train, X_test, y_train, y_test, y_pred, train_indx, test_indx, history = return_dict[-1]
        
        r1,r2,r3,r4,r5 = test_model(y_test, y_pred, name=hyper_params['embedding_file']) #threshold=get_best_threshold(y_test, y_pred)
        f1.append(r1);acc.append(r2);auc.append(r3);preci.append(r4);recall.append(r5)


    # f = lambda x,y: np.mean(np.array(x)[:,y])
    f = lambda x,y: np.mean(np.array(x), axis=0)[y]
    print('\n\n##### Cross Validation result:  #######\n')
    for i in range(len(labels)):
        print('Label {}:'.format(labels[i]))
        print('Average f1_score: %.3f'%f(f1,i))
        print('Average accuracy_score: %.3f'%f(acc,i))
        print('Average auc_score: %.3f'%f(auc,i))
        print('Average precision_score: %.3f'%f(preci,i))
        print('Average recall_score: %.3f'%f(recall,i))

    add_avg = lambda x: list( np.mean(np.array(x), axis=0) ) + ['-----', np.mean(x)]
    markdown_output(xheader=['accuracy', 'precision', 'recall', 'auc', 'f1'], yheader=list(labels) + ['-----', 'Average'],\
        matrix=np.array([add_avg(acc),add_avg(preci),add_avg(recall),add_avg(auc),add_avg(f1)]).T,\
        table_name="\n**{}   k-fold:{}  embedding shape:{} \nrate:{} batch_size:{} epochs:{}**".format(hyper_params['model_name'],\
            hyper_params['k_fold'], hyper_params['input_shape'], hyper_params['learning_rate'], hyper_params['batch_size'], hyper_params['epochs']))

    return model, X_train, X_test, y_train, y_test, y_pred, train_indx, test_indx, history



def build_model_kfold_ray(Y, hyper_params, metrics=METRICS):
    import pickle
    # bert_default_32.pickle, bert_no_pooling_32.pickle
    X = read_pickle(hyper_params['embedding_file'])

    f1,acc,auc,preci,recall = [],[],[],[],[]
    from sklearn.model_selection import train_test_split

    ray.init()
    futures = [build_model_kfold_train.remote(X, Y, hyper_params, metrics) for i in range(hyper_params['k_fold'])]
    return_dict = ray.get(futures)

    for i in range(hyper_params['k_fold']):
        
        model, X_train, X_test, y_train, y_test, y_pred, train_indx, test_indx, history = return_dict[i]
        
        r1,r2,r3,r4,r5 = test_model(y_test, y_pred, name=hyper_params['embedding_file']) #threshold=get_best_threshold(y_test, y_pred)
        f1.append(r1);acc.append(r2);auc.append(r3);preci.append(r4);recall.append(r5)


    # f = lambda x,y: np.mean(np.array(x)[:,y])
    f = lambda x,y: np.mean(np.array(x), axis=0)[y]
    print('\n\n##### Cross Validation result:  #######\n')
    for i in range(len(labels)):
        print('Label {}:'.format(labels[i]))
        print('Average f1_score: %.3f'%f(f1,i))
        print('Average accuracy_score: %.3f'%f(acc,i))
        print('Average auc_score: %.3f'%f(auc,i))
        print('Average precision_score: %.3f'%f(preci,i))
        print('Average recall_score: %.3f'%f(recall,i))

    add_avg = lambda x: list( np.mean(np.array(x), axis=0) ) + ['-----', np.mean(x)]
    markdown_output(xheader=['accuracy', 'precision', 'recall', 'auc', 'f1'], yheader=list(labels) + ['-----', 'Average'],\
        matrix=np.array([add_avg(acc),add_avg(preci),add_avg(recall),add_avg(auc),add_avg(f1)]).T,\
        table_name="\n**{}   k-fold:{}  embedding shape:{} \nrate:{} batch_size:{} epochs:{}**".format(hyper_params['model_name'],\
            hyper_params['k_fold'], hyper_params['input_shape'], hyper_params['learning_rate'], hyper_params['batch_size'], hyper_params['epochs']))

    return model, X_train, X_test, y_train, y_test, y_pred, train_indx, test_indx, history




def build_model_kfold_bk(Y, hyper_params, metrics=METRICS):
    import pickle
    # bert_default_32.pickle, bert_no_pooling_32.pickle
    X = read_pickle(hyper_params['embedding_file'])

    f1,acc,auc,preci,recall = [],[],[],[],[]
    from sklearn.model_selection import train_test_split

    for _ in range(hyper_params['k_fold']):
        X_train, X_test, y_train, y_test, train_indx, test_indx = train_test_split(X, Y, list(range(len(X))), test_size=0.33, random_state=42)
        print('X_train shape: {}\n y_train shape: {}\n X_test shape: {}\ny_test shape: {}'.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

        model = eval(hyper_params['model_name'])(hyper_params)
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
        r1,r2,r3,r4,r5 = test_model(y_test, y_pred, name=hyper_params['embedding_file']) #threshold=get_best_threshold(y_test, y_pred)
        f1.append(r1);acc.append(r2);auc.append(r3);preci.append(r4);recall.append(r5)
        cuda.select_device(0)
        cuda.close()


    # f = lambda x,y: np.mean(np.array(x)[:,y])
    f = lambda x,y: np.mean(np.array(x), axis=0)[y]
    print('\n\n##### Cross Validation result:  #######\n')
    for i in range(len(labels)):
        print('Label {}:'.format(labels[i]))
        print('Average f1_score: %.3f'%f(f1,i))
        print('Average accuracy_score: %.3f'%f(acc,i))
        print('Average auc_score: %.3f'%f(auc,i))
        print('Average precision_score: %.3f'%f(preci,i))
        print('Average recall_score: %.3f'%f(recall,i))

    add_avg = lambda x: list( np.mean(np.array(x), axis=0) ) + ['-----', np.mean(x)]
    markdown_output(xheader=['accuracy', 'precision', 'recall', 'auc', 'f1'], yheader=list(labels) + ['-----', 'Average'],\
        matrix=np.array([add_avg(acc),add_avg(preci),add_avg(recall),add_avg(auc),add_avg(f1)]).T,\
        table_name="\n**{}   k-fold:{}  embedding shape:{} \nrate:{} batch_size:{} epochs:{}**".format(hyper_params['model_name'],\
            hyper_params['k_fold'], hyper_params['input_shape'], hyper_params['learning_rate'], hyper_params['batch_size'], hyper_params['epochs']))

    return model, X_train, X_test, y_train, y_test, y_pred, train_indx, test_indx, history




def build_model_kfold(Y, hyper_params, metrics=METRICS):
    import pickle
    # bert_default_32.pickle, bert_no_pooling_32.pickle
    X = read_pickle(hyper_params['embedding_file'])

    f1,acc,auc,preci,recall = [],[],[],[],[]
    from sklearn.model_selection import train_test_split

    for _ in range(hyper_params['k_fold']):
        X_train, X_test, y_train, y_test, train_indx, test_indx = train_test_split(X, Y, list(range(len(X))), test_size=0.33, random_state=42)
        print('X_train shape: {}\n y_train shape: {}\n X_test shape: {}\ny_test shape: {}'.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

        model = eval(hyper_params['model_name'])(hyper_params)
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
        r1,r2,r3,r4,r5 = test_model(y_test, y_pred, name=hyper_params['embedding_file']) #threshold=get_best_threshold(y_test, y_pred)
        f1.append(r1);acc.append(r2);auc.append(r3);preci.append(r4);recall.append(r5)
        # cuda.select_device(0)
        # cuda.close()


    # f = lambda x,y: np.mean(np.array(x)[:,y])
    f = lambda x,y: np.mean(np.array(x), axis=0)[y]
    print('\n\n##### Cross Validation result:  #######\n')
    for i in range(len(labels)):
        print('Label {}:'.format(labels[i]))
        print('Average f1_score: %.3f'%f(f1,i))
        print('Average accuracy_score: %.3f'%f(acc,i))
        print('Average auc_score: %.3f'%f(auc,i))
        print('Average precision_score: %.3f'%f(preci,i))
        print('Average recall_score: %.3f'%f(recall,i))

    add_avg = lambda x: list( np.mean(np.array(x), axis=0) ) + ['-----', np.mean(x)]
    markdown_output(xheader=['accuracy', 'precision', 'recall', 'auc', 'f1'], yheader=list(labels) + ['-----', 'Average'],\
        matrix=np.array([add_avg(acc),add_avg(preci),add_avg(recall),add_avg(auc),add_avg(f1)]).T,\
        table_name="\n**{}   k-fold:{}  embedding shape:{} \nrate:{} batch_size:{} epochs:{}**".format(hyper_params['model_name'],\
            hyper_params['k_fold'], hyper_params['input_shape'], hyper_params['learning_rate'], hyper_params['batch_size'], hyper_params['epochs']))

    return model, X_train, X_test, y_train, y_test, y_pred, train_indx, test_indx, history



def build_model_kfold_wtest(Y, hyper_params, metrics=METRICS):
    function_name = hyper_params['model_name']
    import pickle
    # bert_default_32.pickle, bert_no_pooling_32.pickle
    X = read_pickle(hyper_params['embedding_file'])


    f1,acc,auc,preci,recall = [],[],[],[],[]
    from sklearn.model_selection import train_test_split

    
    indx,y_test_list,y_pred_list = read_wtest_pickle(function_name+'.pickle')
    kf = KFold(n_splits=10)

    t = -1
    for train_index, test_index in kf.split(X):
        t = t+1
        if t != indx+1:
            continue
        print('Current fold: {}'.format(t))
        # X_train, X_test, y_train, y_test, train_indx, test_indx = train_test_split(X, Y, list(range(len(X))), test_size=0.33, random_state=42)
        X,Y = np.array(X),np.array(Y)
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], Y[train_index], Y[test_index], 
        print('X_train shape: {}\n y_train shape: {}\n X_test shape: {}\ny_test shape: {}'.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

        model = eval(hyper_params['model_name'])(hyper_params)
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

        y_test_list.append(y_test)
        y_pred_list.append(y_pred)
        save_wtest_pickle((t,y_test_list,y_pred_list), function_name+'.pickle')
        if indx!=8:
            print('Current fold: {}'.format(t))
            time.sleep(6)
            exit(0)

    for y_test,y_pred in zip(y_test_list,y_pred_list):
        r1,r2,r3,r4,r5 = test_model(y_test, y_pred, name=hyper_params['embedding_file']) #threshold=get_best_threshold(y_test, y_pred)
        f1.append(r1);acc.append(r2);auc.append(r3);preci.append(r4);recall.append(r5)
        # cuda.select_device(0)
        # cuda.close()

    print('\n\n##### F1 and AUC for wtest:  #######\n')
    print('f1:\n{}\nauc:\n{}\n'.format(f1,auc))

    # f = lambda x,y: np.mean(np.array(x)[:,y])
    f = lambda x,y: np.mean(np.array(x), axis=0)[y]
    print('\n\n##### Cross Validation result:  #######\n')
    for i in range(len(labels)):
        print('Label {}:'.format(labels[i]))
        print('Average f1_score: %.3f'%f(f1,i))
        print('Average accuracy_score: %.3f'%f(acc,i))
        print('Average auc_score: %.3f'%f(auc,i))
        print('Average precision_score: %.3f'%f(preci,i))
        print('Average recall_score: %.3f'%f(recall,i))

    add_avg = lambda x: list( np.mean(np.array(x), axis=0) ) + ['-----', np.mean(x)]
    markdown_output(xheader=['accuracy', 'precision', 'recall', 'auc', 'f1'], yheader=list(labels) + ['-----', 'Average'],\
        matrix=np.array([add_avg(acc),add_avg(preci),add_avg(recall),add_avg(auc),add_avg(f1)]).T,\
        table_name="\n**{}   k-fold:{}  embedding shape:{} \nrate:{} batch_size:{} epochs:{}**".format(hyper_params['model_name'],\
            hyper_params['k_fold'], hyper_params['input_shape'], hyper_params['learning_rate'], hyper_params['batch_size'], hyper_params['epochs']))


    return model, X_train, X_test, y_train, y_test, y_pred, train_index, test_index, history




def build_model_liwc(Y, hyper_params, metrics=METRICS):
    import pickle
    # bert_default_32.pickle, bert_no_pooling_32.pickle
    X = read_pickle(hyper_params['embedding_file'])
    liwc = read_liwc(normalize=True)

    f1,acc,auc,preci,recall = [],[],[],[],[]
    from sklearn.model_selection import train_test_split
    # print('X shape: {}\n y shape: {}'.format(X.shape, Y.shape))

    X_train, X_test, liwc_train, liwc_test, y_train, y_test = train_test_split(X, liwc, Y, test_size=0.33, random_state=42)
    print('X_train shape: {}\n y_train shape: {}\n X_test shape: {}\ny_test shape: {}'.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

    model = eval(hyper_params['model_name'])(hyper_params)
    # for i in model.layers:
    #     if 'trainable' in i.name:
    #         i.trainable = True
    model.compile(optimizer=Adam(lr=hyper_params['learning_rate']), loss=hyper_params['loss'], metrics=metrics) #['acc',f1_m,precision_m, recall_m]
    print(model.summary())
    print('y_train.shape: {}'.format(y_train.shape))
    print([dict(enumerate(class_weight.compute_class_weight('balanced', [0,1], i.squeeze()))) for i in y_train.T])

    if 'asinput' in hyper_params['model_name']:
        history = model.fit(x= [X_train, liwc_train], y=[ i.squeeze() for i in y_train.T], 
                epochs=hyper_params['epochs'], 
                batch_size=hyper_params['batch_size'],
                class_weight=[dict(enumerate(class_weight.compute_class_weight('balanced', [0,1], i.squeeze()))) for i in y_train.T],
                validation_data=([X_test, liwc_test], [ i.squeeze() for i in y_test.T]) # validation_split=0.25
        )
        y_pred = model.predict([X_test, liwc_test])
    elif 'asoutput' in hyper_params['model_name']:
        history = model.fit(x=X_train, y=[ i.squeeze() for i in y_train.T] + [liwc_train], 
                epochs=hyper_params['epochs'], 
                batch_size=hyper_params['batch_size'],
                class_weight=[dict(enumerate(class_weight.compute_class_weight('balanced', [0,1], i.squeeze()))) for i in y_train.T] + [lambda x: x + 1.],
                validation_data=(X_test, [ i.squeeze() for i in y_test.T] + [liwc_test]) # validation_split=0.25
        )
        y_pred = model.predict(X_test )[:-1]
    else:
        assert(False)

    r1,r2,r3,r4,r5 = test_model(y_test, y_pred, name=hyper_params['embedding_file'])
    f1.append(r1);acc.append(r2);auc.append(r3);preci.append(r4);recall.append(r5)
    return model, X_train, X_test, y_train, y_test, y_pred, history


def build_model_liwc_kfold(Y, hyper_params, metrics=METRICS):
    import pickle
    # bert_default_32.pickle, bert_no_pooling_32.pickle
    X = read_pickle(hyper_params['embedding_file'])
    liwc = read_liwc(normalize=True)

    f1,acc,auc,preci,recall = [],[],[],[],[]
    from sklearn.model_selection import train_test_split
    # print('X shape: {}\n y shape: {}'.format(X.shape, Y.shape))
    
    for _ in range(hyper_params['k_fold']):
        X_train, X_test, liwc_train, liwc_test, y_train, y_test = train_test_split(X, liwc, Y, test_size=0.33, random_state=42)
        print('X_train shape: {}\n y_train shape: {}\n X_test shape: {}\ny_test shape: {}'.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

        model = eval(hyper_params['model_name'])(hyper_params)
        for i in model.layers:
            if 'trainable' in i.name:
                i.trainable = True
        model.compile(optimizer=Adam(lr=hyper_params['learning_rate']), loss=hyper_params['loss'], metrics=metrics) #['acc',f1_m,precision_m, recall_m]
        print(model.summary())
        print('y_train.shape: {}'.format(y_train.shape))

        if 'asinput' in hyper_params['model_name']:
            history = model.fit(x= [X_train, liwc_train], y=[ i.squeeze() for i in y_train.T], 
                    epochs=hyper_params['epochs'], 
                    batch_size=hyper_params['batch_size'],
                    class_weight=[dict(enumerate(class_weight.compute_class_weight('balanced', [0,1], i.squeeze()))) for i in y_train.T],
                    validation_data=([X_test, liwc_test], [ i.squeeze() for i in y_test.T]) # validation_split=0.25
            )
            y_pred = model.predict([X_test, liwc_test])
        elif 'asoutput' in hyper_params['model_name']:
            history = model.fit(x=X_train, y=[ i.squeeze() for i in y_train.T] + [liwc_train], 
                    epochs=hyper_params['epochs'], 
                    batch_size=hyper_params['batch_size'],
                    class_weight=[dict(enumerate(class_weight.compute_class_weight('balanced', [0,1], i.squeeze()))) for i in y_train.T] + [lambda x: 1.0],
                    validation_data=(X_test, [ i.squeeze() for i in y_test.T] + [liwc_test]) # validation_split=0.25
            )
            y_pred = model.predict(X_test )[:-1]
        else:
            assert(False)

        r1,r2,r3,r4,r5 = test_model(y_test, y_pred, name=hyper_params['embedding_file'])
        f1.append(r1);acc.append(r2);auc.append(r3);preci.append(r4);recall.append(r5)


    # f = lambda x,y: np.mean(np.array(x)[:,y])
    f = lambda x,y: np.mean(np.array(x), axis=0)[y]
    print('\n\n##### Cross Validation result:  #######\n')
    for i in range(len(labels)):
        print('Label {}:'.format(labels[i]))
        print('Average f1_score: %.3f'%f(f1,i))
        print('Average accuracy_score: %.3f'%f(acc,i))
        print('Average auc_score: %.3f'%f(auc,i))
        print('Average precision_score: %.3f'%f(preci,i))
        print('Average recall_score: %.3f'%f(recall,i))

    add_avg = lambda x: list( np.mean(np.array(x), axis=0) ) + ['-----', np.mean(x)]
    markdown_output(xheader=['accuracy', 'precision', 'recall', 'auc', 'f1'], yheader=list(labels) + ['-----', 'Average'],\
        matrix=np.array([add_avg(acc),add_avg(preci),add_avg(recall),add_avg(auc),add_avg(f1)]).T,\
        table_name="\n**{}   k-fold:{}  embedding shape:{} \nrate:{} batch_size:{} epochs:{}**".format(hyper_params['model_name'],\
            hyper_params['k_fold'], hyper_params['input_shape'], hyper_params['learning_rate'], hyper_params['batch_size'], hyper_params['epochs']))

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
    print('X_train shape: {}\n y_train shape: {}\n X_test shape: {}\ny_test shape: {}'.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

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
    # for i in model.layers:
    #     if 'trainable' in i.name:
    #         i.trainable = True
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



def build_model_triple_kfold(Y, hyper_params, metrics=METRICS):
    import pickle
    # bert_default_32.pickle, bert_no_pooling_32.pickle
    bertx = read_pickle('bert_no_pooling_34_12860x34x1024.pickle')
    elmox = read_pickle('elmo_34_12860x34x1024.pickle')
    glovex = read_pickle('glove_34_12860x34x200.pickle')


    f1,acc,auc,preci,recall = [],[],[],[],[]
    from sklearn.model_selection import train_test_split
    for _ in range(hyper_params['k_fold']):
        bertx_train, bertx_test, elmox_train, elmox_test, glovex_train, glovex_test, y_train, y_test = train_test_split(bertx, elmox, glovex, Y, test_size=0.33, random_state=42)

        model = keras_bert_triple(hyper_params)
        # for i in model.layers:
        #     if 'trainable' in i.name:
        #         i.trainable = True
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

    # f = lambda x,y: np.mean(np.array(x)[:,y])
    f = lambda x,y: np.mean(np.array(x), axis=0)[y]
    print('\n\n##### Cross Validation result:  #######\n')
    for i in range(len(labels)):
        print('Label {}:'.format(labels[i]))
        print('Average f1_score: %.3f'%f(f1,i))
        print('Average accuracy_score: %.3f'%f(acc,i))
        print('Average auc_score: %.3f'%f(auc,i))
        print('Average precision_score: %.3f'%f(preci,i))
        print('Average recall_score: %.3f'%f(recall,i))

    add_avg = lambda x: list( np.mean(np.array(x), axis=0) ) + ['-----', np.mean(x)]
    markdown_output(xheader=['accuracy', 'precision', 'recall', 'auc', 'f1'], yheader=list(labels) + ['-----', 'Average'],\
        matrix=np.array([add_avg(acc),add_avg(preci),add_avg(recall),add_avg(auc),add_avg(f1)]).T,\
        table_name="\n**{}   k-fold:{}  embedding shape:{} \nrate:{} batch_size:{} epochs:{}**".format(hyper_params['model_name'],\
            hyper_params['k_fold'], hyper_params['input_shape'], hyper_params['learning_rate'], hyper_params['batch_size'], hyper_params['epochs']))

    return model, bertx_train, bertx_test, elmox_train, elmox_test, glovex_train, glovex_test, y_train, y_test, y_pred, history

