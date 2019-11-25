#!/usr/local/bin/python3
# -*- coding: utf-8 -*- 

from header import *

def get_embedding_matrix(dimension, t):
    # glove_root_folder = "/Users/reklanirs/workspace/NLP/Project/claff-happydb/"
    f = open(os.path.join(root_folder, 'glove.twitter.27B/glove.twitter.27B.%dd.txt'%dimension), 'r', encoding="utf-8")
    embeddings_index = dict()
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    embedding_matrix = np.zeros((len(t.word_index) + 1, dimension))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            assert(len(embedding_vector) == dimension)
            embedding_matrix[i] = embedding_vector
    return  embedding_matrix
    


def keras_cnn_mtl_emb_glove(hparam):
    input_text = keras.Input(shape=(hparam['maxlen'],))
    x = keras.layers.Embedding(input_dim=hparam['max_words'],
                                             output_dim=hparam['embedding_dim'],
                                             input_length=hparam['maxlen'],
                                             weights=[hparam['weights']],
                                             trainable=hparam['train_able'])(input_text)

    x = NonMasking()(x)
    x = SpatialDropout1D(0.3)(x)
    # conv = keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu')(embedding_layer)
    x = Bidirectional(GRU(100, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])

    output1 = keras.layers.Dense(hparam['T'], activation='sigmoid' if hparam['isDataBinary'] else 'relu')(conc)

    model = keras.Model(inputs=input_text, outputs=output1)
    # print the model summary
    print(model.summary())
    return model
