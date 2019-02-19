import os
import gc
import sys
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import json
import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import *
from keras.layers import *
from keras.utils import to_categorical

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers, optimizers

import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk import tokenize

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing

tknzr = TweetTokenizer()
stop_set = set(stopwords.words('english') + list(string.punctuation))

def preprocessing_tweet(folder, category, all_text, all_time, all_freq, label, reviews):
    count = 0
    for filename in os.listdir(folder):
#        if count > 10:
#            break
#        count +=1
        
        sentences = []
        time = []
        freq = []
        if os.path.isdir(os.path.join(folder, filename)):
            path = folder + '/' + filename + '/tweets.json'
            tweets_data = []
            with open(path, 'r', encoding="utf-8") as f:
                for line in f:
                    try:
                        tweet = json.loads(line)
                        if 'RT @' not in tweet['text']:
                            tweets_data.append(tweet)
                    except:
                        continue
            tweets = pd.DataFrame()
            tweets['text'] = list(map(lambda tweet: tweet.get('text'), tweets_data))
            tweets['time'] = list(map(lambda tweet: tweet.get('created_at'), tweets_data))
            path = folder + '/' + filename + '/concatenate.txt'
            
            with open(path, 'w', encoding="utf-8") as f:
                for i in range(len(tweets)):
                    
                    freq.append(int(tweets['time'][i].split()[2]))
                    time.append(int(tweets['time'][i].split()[3].split(":")[0]))
                    
                    tokens = tknzr.tokenize(tweets['text'][i].replace('\n', ' ;'))
                    if tokens[0] == 'rt':
                        tokens = tokens[1:]
                        
                    tweet = '';
                    for x in tokens:
                        if len(x) > 0 and x[0] != '@' and 'http' not in x and 'diagnosed' not in x:
                            tweet = tweet + ' ' + x;
                    f.write(tweet +' ')
                    sentences.append(tweet)
            with open(path, 'r', encoding="utf-8") as f:
                data = f.read()
                
            buckets = [0] * 31
            freq = [x - min(freq) for x in freq]
            for x in freq:
                buckets[x] += 1
                
            all_text.append(data)
            label.append(category)
    #        sentences = tokenize.sent_tokenize(data)
            reviews.append(sentences)
            all_time.append(time)
            all_freq.append(buckets)



MAX_SENT_LENGTH = 100
MAX_SENTS = 150
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.25
filter_sizes = [3,4,5]
num_filters = 450
sequence_length = 3000


count = 0
all_text = []
all_time = []
all_freq = []
label = []
reviews = []
            
pos = './label/positive'
preprocessing_tweet(pos, 1, all_text, all_time, all_freq, label, reviews)
neg = './label/negative'
preprocessing_tweet(neg, 0, all_text, all_time, all_freq, label, reviews)

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
#tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:<=>?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(all_text)

data = np.zeros((len(all_text), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
for i, sentences in enumerate(reviews):
    for j, sent in enumerate(sentences):
        if j < MAX_SENTS:
            wordTokens = text_to_word_sequence(sent)
            k = 0
            for _, word in enumerate(wordTokens):
                if k < MAX_SENT_LENGTH and tokenizer.word_index[word] < MAX_NB_WORDS:
                    data[i, j, k] = tokenizer.word_index[word]
                    k = k + 1

vocab = tokenizer.word_index

#sequences = tokenizer.texts_to_matrix(all_text, mode='binary')
sequences = tokenizer.texts_to_sequences(all_text)
sequences = pad_sequences(sequences, maxlen=sequence_length)
print('Shape of data tensor:', len(data))
print('Shape of label tensor:', len(label))
label = to_categorical(label)

#b = np.zeros([len(all_time),len(max(all_time,key = lambda x: len(x)))])
#for i,j in enumerate(all_time):
#    b[i][0:len(j)] = j
#
##b = np.concatenate((b, np.asarray(all_freq)), axis=1)
#
#b = preprocessing.scale(b)
#
#c = np.asarray(all_freq)
#c = preprocessing.scale(c)


GLOVE_DIR = './Glove/' + 'glove.twitter.27B.'+str(EMBEDDING_DIM)+'d.txt'

embeddings_index = {}
with open(GLOVE_DIR, 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')

embedding_matrix = np.random.random((len(vocab) + 1, EMBEDDING_DIM))
for word, i in vocab.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


def dot_product(x, kernel):

    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class AttLayer(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True,
                 return_attention=False,
                 **kwargs):

        self.supports_masking = True
        self.return_attention = return_attention
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        eij = dot_product(x, self.W)

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        weighted_input = x * K.expand_dims(a)

        result = K.sum(weighted_input, axis=1)

        if self.return_attention:
            return [result, a]
        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], input_shape[-1]),
                    (input_shape[0], input_shape[1])]
        else:
            return input_shape[0], input_shape[-1]


GRU1_unit = 110
Att1_unit = 110
GRU2_unit = 110
Att2_unit = 90

print("model fitting - Hierachical attention network")
print(EMBEDDING_DIM,' Dim,', GRU1_unit, ' ', Att1_unit, ' ', GRU2_unit, ' ',Att2_unit)

np.random.seed(0)
indices = np.arange(data.shape[0])

accumulate = 0
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.1, random_state=1000)
for train_index, test_index in sss.split(sequences, label):

    seq_train, seq_val = sequences[train_index], sequences[test_index]
    x_train, x_val = data[train_index], data[test_index]
    y_train, y_val = label[train_index], label[test_index]
#    t_train, t_val = b[train_index], b[test_index]
#    f_train, f_val = c[train_index], c[test_index]
    
    embedding_layer = Embedding(len(vocab) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SENT_LENGTH,
                            trainable=False)

    sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32' , name='main_input')
    embedded_sequences = embedding_layer(sentence_input)
    l_lstm = Bidirectional(GRU(GRU1_unit, return_sequences=True, dropout = 0.1))(embedded_sequences)
#    print(l_lstm._keras_shape)

    
    l_att = AttLayer()(l_lstm)
    sentEncoder = Model(sentence_input, l_att)

    review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = TimeDistributed(Dense(GRU2_unit))(review_encoder)
#    l_lstm_sent = Dropout(0.1)(l_lstm_sent)
    l_att_sent = AttLayer()(l_lstm_sent)


#    TimeEmbedding_layer = Embedding(25,
#                            60,
#                            input_length=200,
#                            trainable=True)
#    
#    TimeEmbedded_input = Input(shape=(len(b[0]),), name='time_input')
#    TimeEmbedded_output = TimeEmbedding_layer(TimeEmbedded_input)
#    TimeEmbedded_output = Flatten()(TimeEmbedded_output)
#    
#    
#    bucket_input = Input(shape=(len(c[0]),), name='bucket_input')
#    time_output = concatenate([TimeEmbedded_output, bucket_input])
#    time_output = Dense(80)(time_input)
    
    seq_embedding_layer = Embedding(len(vocab) + 1,
                        EMBEDDING_DIM,
                        weights=[embedding_matrix],
                        input_length=sequence_length,
                        trainable=False)
    
    inputs = Input(shape=(sequence_length,), dtype='int32')
    embedding = seq_embedding_layer(inputs)
    reshape = Reshape((sequence_length,EMBEDDING_DIM,1))(embedding)
    
    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], EMBEDDING_DIM), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], EMBEDDING_DIM), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], EMBEDDING_DIM), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
    
    maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)
    
    concatenated_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    
#    l_att_sent = concatenate([l_att_sent, time_output, flatten])
    
    l_att_sent = concatenate([l_att_sent, flatten])
    l_att_sent = Dropout(0.4)(l_att_sent)
    preds = Dense(2, activation='softmax')(l_att_sent)
    model = Model([review_input, inputs], preds)
    
#    model = Model([review_input, TimeEmbedded_input, bucket_input, inputs], preds)
#    model = Model(review_input, preds)
    
    opt = optimizers.RMSprop(lr=0.0005)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['acc'])
    
    
    history = model.fit([x_train, seq_train], y_train, validation_data=([x_val, seq_val], y_val),
          epochs=40,verbose=1, batch_size=16)

#    history = model.fit([x_train, t_train, f_train, seq_train], y_train, validation_data=([x_val, t_val, f_val, seq_val], y_val),
#              epochs=50,verbose=1, batch_size=32)
    accumulate  = accumulate + history.history['val_acc'][-1]
    
    del model
    del history
    gc.collect()

    
print(accumulate/10)
