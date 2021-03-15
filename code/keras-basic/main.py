#!/usr/local/bin/python3
# -*- coding: utf-8 -*- 
from header import *

from file_reader import *
from evaluate import *

from build_model import *

training_data = "data/training data/labeled_training_set.csv"
test_data = "data/test data/unlabeled_test_set.csv"
labeled_test_data = "data/test data/labeled_test_set.txt"

training_data_path = os.path.join(root_folder, training_data)
test_data_path = os.path.join(root_folder, test_data)

systemrun_path = os.path.join(root_folder, 'systemrun')
output_path = os.path.join(root_folder, 'data/output')


df = []
_DEBUG = False


hyper_params_bert = {
    'generate_result': False,
    'test_testdata': True,
    'tricky_threshold': False,
    'learning_rate': 2e-5,
    'maxlen': 34,
    'epochs': 50,
    'batch_size': 256, # 8616

    'drop_rate' : 0.2,
    'T': len(labels),
    'optimizer': 'adam',
    'loss': 'binary_crossentropy', #binary_crossentropy mean_squared_error
    'activation': 'leaky_relu', #  leaky_relu prelu gelu
    'k_fold': 5,
    'w_test': False,
    'embedding_file' : 'bert_default_34_12860x1024.pickle', # bert_default_32_12860x768.pickle shape=(768,), bert_no_pooling_32_12860x32x768.pickle shape=(32,768)
                                                                # bert_default_32_12860x768.pickle, bert_no_pooling_32_12860x32x768.pickle, 
                                                                # bert_default_34_12860x1024.pickle, bert_no_pooling_34_12860x34x1024.pickle
                                                                # bert_no_pooling_44_12860x44x1024.pickle
    'input_shape': (),
    'model_name': 'keras_bert_bk', # keras_bert_bk,  keras_bert_prelu, keras_bert_triple, keras_bert_liwc_asinput, keras_bert_liwc_asoutput, keras_bert_group1, keras_bert_group_venn, keras_bert_group_venn2
                                            #                 keras_bert_prelu,                    keras_bert_liwc_asinput, keras_bert_liwc_asoutput, keras_bert_group1
}

embedding_file_name_2_input_shape = lambda x: tuple( map(int, x.strip().split('.')[0].split('_')[-1].split('x')[1:]) )
hyper_params_bert['input_shape'] = embedding_file_name_2_input_shape(hyper_params_bert['embedding_file'])

def tmp():
    X, Y = read_data(training_data_path, clean=False)
    X_test, y_test, y_pred, history = build_model_liwc(Y, hyper_params_bert, metrics=[precision_m, recall_m, f1_m,])
    return X_test, y_test, y_pred, history

def manual_test(sentences,y_true,y_pred):
    threshold = get_best_threshold(y_true, y_pred)
    reformat = lambda x: np.array([1 if i>threshold[indx] else 0  for indx,i in enumerate(x)])
    y_true, y_pred = np.array(y_true).squeeze(), np.array(y_pred).squeeze()
    if y_true.shape[0]!=len(sentences):
        y_true = y_true.T
    if y_pred.shape[0]!=len(sentences):
        y_pred = y_pred.T
    for s,i,j in zip(sentences,y_true,y_pred):
        i,j = reformat(i),reformat(j)
        if all(i==j):
            continue
        print('\n'+s)
        for k in range(len(labels)):
            if i[k]!=j[k]:
                print('\tWrong: {}  Real:{} Pred: {}'.format(labels[k], i[k], j[k]))
        tmp = input()
        if tmp == 'fin':
            break
    pass

def generate_result(model, ids, texts, threshold=[0.5]*len(labels)):
    x = read_pickle(hyper_params_bert['embedding_file'].replace('12860', '5000'))
    binary_check = lambda thshld,x: 1 if x>thshld else 0
    binary_format = lambda thshld,y: np.array([ binary_check(thshld,i) for i in y.squeeze() ])
    pred = np.array(model.predict(x)).squeeze()
    if pred.shape[0] != len(labels):
        pred = pred.T
    pred = np.array([binary_format(threshold[i], pred[i]) for i in range(len(labels))]).T

    fpath = os.path.join(output_path, str(datetime.datetime.now().strftime("%Y%m%d-%H%M")))
    tmp = input(fpath)
    if len(tmp)>=1 and tmp[-1] == 'n':
        return
    with open(fpath+'.csv', 'w') as fout:
        for i in range(len(ids)):
            tmp = '{},'.format(ids[i]) + ','.join(map(str,pred[i])) + '\n'
            fout.write(tmp)
    with open(fpath+'_withtext.csv', 'w') as fout:
        for i in range(len(ids)):
            tmp = '{},'.format(ids[i]) + ','.join(map(str,pred[i])) + ',{}\n'.format(texts[i].strip())
            fout.write(tmp)
    print('Output finished.')

def test_testdata(model, ids, texts):
    df = pd.read_csv(os.path.join(root_folder, labeled_test_data))
    ret = np.array(extract_columns(df, labeled_testset_labels))
    dic = {}
    for i in range(len(ret[0])):
        dic[int(ret[0][i])] = list(map(int,ret[2:,i]))
    test_label = np.array([ dic[i] for i in ids])
    pass
    if hyper_params_bert["model_name"].endswith("triple"):
        x = [read_pickle('bert_no_pooling_34_5000x34x1024.pickle'),read_pickle('elmo_34_5000x34x1024.pickle'),read_pickle('glove_34_5000x34x200.pickle')]
    elif 'liwc' in hyper_params_bert["model_name"]:
        if 'asinput' in hyper_params_bert['model_name']:
            x = [read_pickle(hyper_params_bert["embedding_file"].replace("12860", "5000")), read_liwc('/Users/reklanirs/Dropbox/Experiment/claff-offmychest/data/test data/LIWC2015 Results (unlabeled_test_set.csv).csv')]
        elif 'asoutput' in hyper_params_bert['model_name']:
            x = read_pickle(hyper_params_bert["embedding_file"].replace("12860", "5000"))
    else:
        x = read_pickle(hyper_params_bert["embedding_file"].replace("12860", "5000"))
    binary_check = lambda thshld,x: 1 if x>thshld else 0
    binary_format = lambda thshld,y: np.array([ binary_check(thshld,i) for i in y.squeeze() ])
    pred = np.array(model.predict(x)).squeeze() if not 'keras_bert_liwc_asoutput' in hyper_params_bert["model_name"] else np.array(rett[:-1]).squeeze()
    if pred.shape[0] != len(labels):
        pred = pred.T
    if len(labels) == len(all_labels):
        pred = pred[[0,1,5,4]]
    # get ['Emotional_disclosure','Information_disclosure','Emo_support','Info_support']
    # from ['Emotional_disclosure','Information_disclosure','Support','General_support','Info_support','Emo_support']
    test_model(test_label[:,labeled_testset_labels.index(labels[0])-6] if len(labels)==1 else test_label[:,0:4], pred, \
        name="Shared Task label version", threshold=None, labels=labels if len(labels)==1 else ['Emotional_disclosure','Information_disclosure','Emo_support','Info_support'])
    test_model(test_label[:,labeled_testset_labels.index(labels[0])-2] if len(labels)==1 else test_label[:,4:], pred, \
        name="Better label version", threshold=None, labels=labels if len(labels)==1 else ['Emotional_disclosure','Information_disclosure','Emo_support','Info_support'])
    pass


def main():
    if len(sys.argv) > 1:
        hyper_params_bert['epochs'] = int(sys.argv[-1])

    X, Y = read_data(training_data_path, clean=False)

    if hyper_params_bert['model_name'] == 'keras_bert_triple':
        build_model_name = 'build_model_triple_kfold' if hyper_params_bert['k_fold'] is not False else 'build_model_triple'
    elif hyper_params_bert['model_name'].startswith('keras_bert_liwc'):
        build_model_name = 'build_model_liwc_kfold' if hyper_params_bert['k_fold'] is not False else 'build_model_liwc'
    else:
        build_model_name = 'build_model_kfold' if hyper_params_bert['k_fold'] is not False else 'build_model_bert'

    if hyper_params_bert['w_test']:
        build_model_name = build_model_name + '_wtest'

    ret = eval(build_model_name)(Y, hyper_params_bert, metrics=[precision_m, recall_m, f1_m,])
    # model, X_train, X_test, y_train, y_test, y_pred, train_indx, test_indx, history = ret
    model = ret[0]
    history = ret[-1]

    # model, bertx_train, bertx_test, elmox_train, elmox_test, glovex_train, glovex_test, y_train, y_test, y_pred, history = build_model_triple(Y, hyper_params_bert, metrics=[precision_m, recall_m, f1_m,]) 
    # manual_test(np.array([X[i] for i in test_indx]), y_test, y_pred)
    from ploting import plot_metrics
    plot_metrics(history)


    if hyper_params_bert['tricky_threshold'] == True:
        threshold=get_best_threshold(y_test, y_pred)
        test_model(y_test, y_pred, threshold=threshold) #threshold=get_best_threshold(y_test, y_pred)

    if hyper_params_bert['generate_result'] == True:
        final_test_sentenceid, final_test_text = extract_columns(pd.read_csv(test_data_path, header=0), ['sentenceid', 'full_text'])
        generate_result(model, final_test_sentenceid, final_test_text, threshold=threshold)

    if hyper_params_bert['test_testdata'] == True:
        print('\n---\n## {}'.format(hyper_params_bert['model_name']))
        final_test_sentenceid, final_test_text = extract_columns(pd.read_csv(test_data_path, header=0), ['sentenceid', 'full_text'])
        test_testdata(model, final_test_sentenceid, final_test_text)

    # return model, X_train, X_test, y_train, y_test, y_pred, train_indx, test_indx, history
    pass


if __name__ == '__main__':
    main()




