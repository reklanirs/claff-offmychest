#!/usr/local/bin/python3
# -*- coding: utf-8 -*- 

from header import *

def extract_columns(df, l):
    # print(df.iloc[0])
    tmp = {}
    for i in l:
        tmp[i] = np.array(df[i].tolist())
    ret = [tmp[i] for i in l]
    return ret

def read_data(data_path, clean=True, DEBUG = False):
    print('\n## read_data from {}'.format(data_path))
    df = pd.read_csv(data_path, header=0)
    print('Total number of data: {}'.format(len(df)))
    if DEBUG:
        for i in labels:
            print('%s distribution:\n{}'.format(df[i].value_counts().sort_index()))
    ret = extract_columns(df, ['full_text'] + labels)
    X,Y = np.array(ret[0]), np.array(ret[1:]).transpose()
    if clean:
        X,Y = clean_data(X,Y)
    return X,Y

def read_pickle(filename):
    with open( os.path.join(root_folder, 'data/pickle/', filename),"rb") as pickle_in:
        ret = pickle.load(pickle_in)
    return ret

'''
def read_hyper_params(filename):
    with open(filename, 'r') as f:
        datastore = json.load(f)
    return datastore
    pass
'''

def check_sentence(s):
    if len(s.strip())<=1:
        return False
    # if not all([not i.isalpha() for i in s]):  # exist an alpha character in s
        # return False
    if len(text_to_word_sequence(s)) == 0:
        return False
    return True


def clean_data(X,Y):
    print('\n## clean_data')
    x, y = [], []
    cleaned = 0
    print('Number of data before cleaning: {}'.format(min(len(X), len(Y))))
    for i in range(min(len(X), len(Y))):
        sentence = X[i].strip()
        if check_sentence(sentence):
            x.append(sentence)
            y.append(Y[i])
        else:
            cleaned += 1
    print('Cleaned data number: {}\nNumber of data after cleaning: {}\n'.format(cleaned, len(x)))
    return x,y