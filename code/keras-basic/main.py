#!/usr/local/bin/python3
# -*- coding: utf-8 -*- 
from header import *

from file_reader import *
from evaluate import *

from build_model import *

data_name = "data/training data/labeled_training_set.csv"
data_path = os.path.join(root_folder, data_name)

systemrun_path = os.path.join(root_folder, 'systemrun')



df = []
_DEBUG = False

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


hyper_params_bert = {
    'learning_rate': 1e-4,
    'maxlen': 34,
    'epochs': 30,
    'batch_size': 4096, # 8616
    'T': len(labels),
    'optimizer': 'adam',
    'loss': 'binary_crossentropy',
    'activation': 'leaky_relu', #  leaky_relu prelu
    'embedding_file' : 'bert_no_pooling_34_12860x34x1024.pickle', # bert_default_32_12860x768.pickle shape=(768,), bert_no_pooling_32_12860x32x768.pickle shape=(32,768)
                                                                # bert_default_32_12860x768.pickle, bert_no_pooling_32_12860x32x768.pickle, bert_default_34_12860x1024.pickle, bert_no_pooling_34_12860x34x1024.pickle
    'input_shape': ()
}
embedding_file_name_2_input_shape = lambda x: tuple( map(int, x.strip().split('.')[0].split('_')[-1].split('x')[1:]) )
hyper_params_bert['input_shape'] = embedding_file_name_2_input_shape(hyper_params_bert['embedding_file'])

def tmp():
    X, Y = read_data(data_path, clean=False)
    X_test, y_test, y_pred, history = build_model_bert(Y, hyper_params_bert, metrics=[precision_m, recall_m, f1_m,])
    return X_test, y_test, y_pred, history


def main():
    X, Y = read_data(data_path, clean=False)

    # model, X_test, y_test, y_pred, history = build_model(X, Y, model_name = 'keras_elmo', hyper_params=hyper_params, metrics=['accuracy',precision_m, recall_m, f1_m,])

    model, X_train, X_test, y_train, y_test, y_pred, history = build_model_bert(Y, hyper_params_bert, metrics=[precision_m, recall_m, f1_m,])

    from ploting import plot_metrics
    plot_metrics(history)
    return X_test, y_test, y_pred, history

    # try:
    #     evaluate2(y_test, y_pred)
    # except Exception as e:
    #     print(e)
    # finally:
    #     pass
    # return model, X_test, y_test, y_pred


if __name__ == '__main__':
    main()




