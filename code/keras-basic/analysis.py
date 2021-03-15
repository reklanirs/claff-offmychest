#!/usr/local/bin/python3
# -*- coding: utf-8 -*- 

from header import *

from scipy.stats import chi2_contingency
from scipy.stats import chisquare


def obs_matrix(l1,l2):
    assert(len(l1) == len(l2))
    ret = np.array([[0,0],[0,0]])
    for i in range(len(l1)):
        ret[l1[i]][l2[i]] += 1
    return ret

def chi2_test(y, thredhold=10):
    y = np.array(y).squeeze()
    order = []
    if y.shape[0] != len(labels):
        y = y.T
    for i in range(len(labels)-1):
        for j in range(i+1, len(labels)):
            obs = obs_matrix(y[i], y[j])
            chi2, p, dof, expctd = chi2_contingency(obs)
            if p > 0.05:
                continue
            print('#### {}, {}\nchi2:{}; p:{}; >{}:{}\n'.format(labels[i], labels[j], chi2, p, thredhold, (np.array(obs)>thredhold).all() ))
            order.append((labels[i], labels[j], chi2, p, str((np.array(obs)>thredhold).all())) )
    order.sort(key=lambda x:x[2], reverse=True)
    # print('label1\tlabel2: \t\tchi2\tp')
    # for label1,label2,chi2, p in order:
    #     print('{}\t{}: \t\t{}\t{}'.format(label1,label2,chi2,p))
    markdown_output(['label1','label2','chi2','p', ' >%d'%thredhold], list(range(len(order))), order, 'chi2_test')
    return order
