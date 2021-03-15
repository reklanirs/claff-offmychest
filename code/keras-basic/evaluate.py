#!/usr/local/bin/python3
# -*- coding: utf-8 -*- 

from header import *


def general_evaluate(y_true, y_pred):
    mae, mse = [],[]
    # y_true,y_pred = [y_true] if len(y_true.shape) == 1 else np.array(y_true).transpose() , np.array(y_pred).transpose()
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    for label, i,j in zip(labels, y_true, y_pred):
        print('\nLabel \"{}\":'.format(label))
        mae.append(mean_absolute_error(i, j))
        print('Mean_absolute_error: %.3f'%mae[-1])
        mse.append(mean_squared_error(i, j))
        print('Mean_squared_error: %.3f'%mse[-1])
    print('\n<------------------------------------------------>\n')
    print('Average MAE: %.3f'%np.mean(mae))
    print('Average MSE: %.3f'%np.mean(mse))
    return


def binary_evaluate(y_true, y_pred, isDataBinary=False, thredhold=1, labels=labels):
    # f1_score, accuracy_score, precision_score, recall_score, classification_report, roc_auc_score
    f1,acc,auc,preci,recall = [],[],[],[],[]
    if isDataBinary:
        binary_check = lambda x: np.int(np.round( np.squeeze(x) )) #? Do I need int()?
    else:
        binary_check = lambda x: 1 if np.round( np.squeeze(x) ) > (thredhold-1.0)*2.0 else 0
    binary_format = lambda y: np.array([ binary_check(i) for i in y ])
    # y_true,y_pred = [y_true] if len(y_true.shape) == 1 else np.array(y_true).transpose() , np.array(y_pred).transpose()
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    for label, i,j in zip(labels, y_true, y_pred):
        print('\nLabel \"{}\":'.format(label))
        # auc.append(roc_auc_score(i,j))
        # print('auc_score: %.3f'%auc[-1])

        i,j = binary_format(i), binary_format(j) #?
        showinfo(i)
        showinfo(j)
        f1.append(f1_score(i, j))
        print('f1_score: %.3f'%f1[-1])
        acc.append(accuracy_score(i,j))
        print('accuracy_score: %.3f'%acc[-1])
        preci.append(precision_score(i,j))
        print('precision_score: %.3f'%preci[-1])
        recall.append(recall_score(i,j))
        print('recall_score: %.3f'%recall[-1])
    print('\n<------------------------------------------------>\n')
    print('Average f1_score: %.3f'%np.mean(f1))
    print('Average accuracy_score: %.3f'%np.mean(acc))
    print('Average precision_score: %.3f'%np.mean(preci))
    print('Average recall_score: %.3f'%np.mean(recall))
    return


def evaluate(model, hyper_params, X_test, y_test, y_pred):
    binary_evaluate(y_test, y_pred, hyper_params['isDataBinary'], thredhold=hyper_params['thredhold'])

    print('\n<------------------------------------------------>\n')

    if not hyper_params['isDataBinary']:
        general_evaluate(y_test, y_pred)

    print('y_pred: ')
    showinfo(y_pred)
    pass



def showinfo(d):
    d = np.squeeze(np.array(d))
    if len(d.shape) == 1:
        unique, counts = np.unique(d, return_counts=True)
        print(dict(zip(unique, counts)))
    elif len(d.shape) == 2:
        for i in np.round(np.squeeze(np.array(d))):
            unique, counts = np.unique(i, return_counts=True)
            print(dict(zip(unique, counts)))
    pass


def evaluate2(y_true, y_pred):
    # loss, acc = model.evaluate(X, y)
    # print("loss and accuracy on test data: loss = {}, accuracy = {}".format(loss, acc))

    # f1_score, accuracy_score, precision_score, recall_score, classification_report, roc_auc_score
    f1,auc,acc,preci,recall = [],[],[],[],[]
    binary_check = lambda x: round(x)
    binary_format = lambda y: np.array([ binary_check(i) for i in y ])

    y_true,y_pred = np.array(y_true).transpose(), np.array(y_pred).transpose()

    y_true, y_pred = np.array(y_true), np.array(y_pred)
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
    print('Average f1_score: %.3f'%np.mean(f1))
    print('Average accuracy_score: %.3f'%np.mean(acc))
    print('Average precision_score: %.3f'%np.mean(preci))
    print('Average recall_score: %.3f'%np.mean(recall))
    return f1,acc,auc,preci,recall