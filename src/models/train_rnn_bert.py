import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from keras import Input, Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, Reshape, concatenate, BatchNormalization, TimeDistributed, Lambda, Activation, LSTM, Flatten, Convolution1D, GRU, MaxPooling1D
from keras.layers import Bidirectional, InputLayer, SimpleRNN
from keras.constraints import maxnorm
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
import h5py
from sklearn.metrics import classification_report , precision_recall_fscore_support,precision_score,recall_score,f1_score


def artcile_split_and_pred_para(sentences, predictions):
    'Retruns article splitted along with their sentences labels'
    predictions.insert(0,'0\n')
    article, article_splits = [], []
    for i, (sent, pred) in enumerate(zip(sentences, predictions)):
        if sent == 'ARTICLE_SPLIT_LINE\t0\n':
            article_splits.append(article)
            article = []
        else:
            article.append(( sent.split('\t')[0], int(pred.rstrip()) ))
    return article_splits
def article_sent_labels_emb(article_sent_labels):
    X = []
    for _, label in article_sent_labels:
        X.append(label)
    return X
def prepare_data(articles, sentences, predictions , mode='not_categorical'):
    arg = artcile_split_and_pred_para(sentences, predictions)
    features, labels = [], []
    for article_arg, article_lable in zip(arg, articles):
        features.append(article_sent_labels_emb(article_arg))
        labels.append(int(article_lable.split('\t')[1].rstrip()))
    if mode == 'categorical':
        return np.array(features), np_utils.to_categorical(labels)
    else: # mode == 'not_categorial'
        return np.array(features), np.array(labels)


def get_desired_layer_as_feature(embeddings, layers, get_sum=True):
    '''Returns the selected layer as features from a given embeddings object of the top layers of [CLS] embeddings
        layers : list of ids in [1,2,3,4], with a max length of 4
        get_sum : True if sum, set to False if average of layers is desired'''
    features = []
    for item in embeddings:
        CLS_sum = np.zeros(768)
        for layer in layers:
            CLS_sum += item['features'][0]['layers'][layer-1]['values']
        if get_sum: # sum of layers
            features.append(CLS_sum)
        else: # average of layers
            features.append(CLS_sum/len(layers))
    return features




train_articles = open('data/train.tsv').readlines()
train_sent = open('data/train_sent_fixed/dev.tsv').readlines()
train_pred = open('data/train_sent_fixed/predictions_editorial-claim-premise-bert.txt').readlines()

dev_articles = open('data/dev.tsv').readlines()
dev_sent = open('data/dev_sent_fixed/dev.tsv').readlines()
dev_pred = open('../coling/data/dev_sent_fixed/predictions_editorial-claim-premise-bert.txt').readlines()

test_articles = open('data/test/dev.tsv').readlines()
test_sent = open('data/test_sent/dev.tsv').readlines()
test_pred = open('data/test_sent/predictions_editorial-claim-premise-bert.txt').readlines()


train_features, train_labels_categorical = prepare_data(train_articles, train_sent, train_pred, mode='categorical')
dev_features, dev_labels_categorical = prepare_data(dev_articles, dev_sent, dev_pred, mode='categorical')
test_features, test_labels_categorical = prepare_data(test_articles, test_sent, test_pred, mode='categorical')


_, train_labels = prepare_data(train_articles, train_sent, train_pred)
_, dev_labels = prepare_data(dev_articles, dev_sent, dev_pred)
_, test_labels = prepare_data(test_articles, test_sent, test_pred)


train_embeddings = pickle.load(open('data/train.pkl','rb'))
dev_embeddings = pickle.load(open('data/dev.pkl','rb'))
test_embeddings = pickle.load(open('data/test/test.pkl','rb'))

train_embeddings_features = np.array(get_desired_layer_as_feature(train_embeddings, [4]))
dev_embeddings_features = np.array(get_desired_layer_as_feature(dev_embeddings, [4]))
test_embeddings_features = np.array(get_desired_layer_as_feature(test_embeddings, [4]))

print(train_embeddings_features[0].shape, type(train_embeddings_features))
print(dev_embeddings_features[0].shape, type(dev_embeddings_features))
print(test_embeddings_features[0].shape, type(test_embeddings_features))


batch_size = 32


#-------------------------------
from sklearn.utils import shuffle


shuf_dev_emb, shuf_dev_features, shuf_dev_labels = shuffle(dev_embeddings_features, dev_features, list(dev_labels), random_state=293)
dev_count, dev_limit = 0,353
bal_dev_emb, bal_dev_features, bal_dev_labels = [], [], []


for e,f,l in zip(shuf_dev_emb, shuf_dev_features, shuf_dev_labels):
    if l ==0 and dev_count < dev_limit:
        bal_dev_emb.append(e)
        bal_dev_features.append(f)
        bal_dev_labels.append(l)
        dev_count += 1
    elif l==1:
        bal_dev_emb.append(e)
        bal_dev_features.append(f)
        bal_dev_labels.append(l)

bal_dev_emb, bal_dev_features = np.array(bal_dev_emb), np.array(bal_dev_features)


shuf_test_emb, shuf_test_features, shuf_test_labels = shuffle(test_embeddings_features, test_features, list(test_labels), random_state=293)
test_count, test_limit = 0, 418
bal_test_emb, bal_test_features, bal_test_labels = [], [], []

for e,f,l in zip(shuf_test_emb, shuf_test_features, shuf_test_labels):
    if l ==0 and test_count < test_limit:
        bal_test_emb.append(e)
        bal_test_features.append(f)
        bal_test_labels.append(l)
        test_count += 1
    elif l==1:
        bal_test_emb.append(e)
        bal_test_features.append(f)
        bal_test_labels.append(l)

bal_test_emb, bal_test_features = np.array(bal_test_emb), np.array(bal_test_features)
#-----------------------------------




'''Start of Seperate Editorial and Letters Datasets'''
dev_set_ids = pickle.load(open('dev_set_ids_edi_let.p','rb'))
dev_set_ids.keys()

dev_features_edi = [f for i,f in enumerate(dev_features) if i in dev_set_ids['editorial_ids'] or i in dev_set_ids['editorial_news_ids']]
dev_emb_features_edi = [f for i,f in enumerate(dev_embeddings_features) if i in dev_set_ids['editorial_ids'] or i in dev_set_ids['editorial_news_ids']]
dev_labels_edi = [l for i,l in enumerate(dev_labels) if i in dev_set_ids['editorial_ids'] or i in dev_set_ids['editorial_news_ids']]

dev_features_let = [f for i,f in enumerate(dev_features) if i in dev_set_ids['letter_ids'] or i in dev_set_ids['letter_news_ids']]
dev_emb_features_let = [f for i,f in enumerate(dev_embeddings_features) if i in dev_set_ids['letter_ids'] or i in dev_set_ids['letter_news_ids']]
dev_labels_let = [l for i,l in enumerate(dev_labels) if i in dev_set_ids['letter_ids'] or i in dev_set_ids['letter_news_ids']]

dev_emb_features_edi, dev_features_edi = np.array(dev_emb_features_edi), np.array(dev_features_edi)
dev_emb_features_let, dev_features_let = np.array(dev_emb_features_let), np.array(dev_features_let)
print(len(dev_emb_features_edi), len(dev_features_edi), len(dev_labels_edi), Counter(dev_labels_edi))
print(len(dev_emb_features_let), len(dev_features_let), len(dev_labels_let), Counter(dev_labels_let))


test_set_ids = pickle.load(open('test_set_ids_edi_let.p','rb'))

test_features_edi = [f for i,f in enumerate(test_features) if i in test_set_ids['editorial_ids'] or i in test_set_ids['editorial_news_ids']]
test_emb_features_edi = [f for i,f in enumerate(test_embeddings_features) if i in test_set_ids['editorial_ids'] or i in test_set_ids['editorial_news_ids']]
test_labels_edi = [l for i,l in enumerate(test_labels) if i in test_set_ids['editorial_ids'] or i in test_set_ids['editorial_news_ids']]

test_features_let = [f for i,f in enumerate(test_features) if i in test_set_ids['other_ids'] or i in test_set_ids['other_news_ids']]
test_emb_features_let = [f for i,f in enumerate(test_embeddings_features) if i in test_set_ids['other_ids'] or i in test_set_ids['other_news_ids']]
test_labels_let = [l for i,l in enumerate(test_labels) if i in test_set_ids['other_ids'] or i in test_set_ids['other_news_ids']]

test_emb_features_edi, test_features_edi = np.array(test_emb_features_edi), np.array(test_features_edi)
test_emb_features_let, test_features_let = np.array(test_emb_features_let), np.array(test_features_let)
print(len(test_emb_features_edi), len(test_features_edi), len(test_labels_edi), Counter(test_labels_edi))
print(len(test_emb_features_let), len(test_features_let), len(test_labels_let), Counter(test_labels_let))
'''End of Seperate Editorial and Letters Datasets'''





# model 1: BERT only  (sanity check)
model = Sequential()
model.add(InputLayer(input_shape=(train_embeddings_features[0].shape[0],)))
model.add(Dense(2,activity_regularizer=l2(0.0001)))
model.add(Activation('softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(train_embeddings_features, train_labels_categorical,
         batch_size=batch_size,
         epochs=5,
         validation_data=(dev_embeddings_features, dev_labels_categorical))

# print('BERT only, dev data')
# predictions = np.argmax(model.predict(dev_embeddings_features), axis=1)
# print(classification_report(dev_labels, predictions))

# print('BERT only, test data')
# predictions = np.argmax(model.predict(test_embeddings_features), axis=1)
# print(classification_report(test_labels, predictions))



# print('BERT only, bal dev data')
# predictions = np.argmax(model.predict(bal_dev_emb), axis=1)
# print(classification_report(bal_dev_labels, predictions))

# print('BERT only, bal test data')
# predictions = np.argmax(model.predict(bal_test_emb), axis=1)
# print(classification_report(bal_test_labels, predictions))



# Start of Seperate Editorial and Letters Experiments
print('BERT only, editorial dev data')
predictions = np.argmax(model.predict(dev_emb_features_edi), axis=1)
print(classification_report(dev_labels_edi, predictions))

print('BERT only, editorial test data')
predictions = np.argmax(model.predict(test_emb_features_edi), axis=1)
print(classification_report(test_labels_edi, predictions))



print('BERT only, letters dev data')
predictions = np.argmax(model.predict(dev_emb_features_let), axis=1)
print(classification_report(dev_labels_let, predictions))

print('BERT only, letters test data')
predictions = np.argmax(model.predict(test_emb_features_let), axis=1)
print(classification_report(test_labels_let, predictions))
# End of Seperate Editorial and Letters Experiments



# model 2: RNN
max_features = 3
maxlen = 80
batch_size = 32

print('Loading data...')
x_train, y_train, x_dev, y_dev = train_features, train_labels_categorical, dev_features, dev_labels_categorical

print(len(x_train), 'train sequences')
print(len(x_dev), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_dev = sequence.pad_sequences(x_dev, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_dev shape:', x_dev.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(SimpleRNN(128, dropout=0.2)) #, recurrent_dropout=0.5\n",
model.add(Dense(2, activation='softmax'))

# try using different optimizers and different optimizer configs\n",
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train, batch_size=batch_size, epochs=5, validation_data=(x_dev, y_dev))

# model.save_weights(\"model_lstm_15ep.h5\")\n",


# predictions = np.argmax(model.predict(x_dev), axis=1)
# print(classification_report(y_dev, predictions))


# print('RNN, dev data')
# x_dev_arg = sequence.pad_sequences(dev_features, maxlen=maxlen)
# predictions = np.argmax(model.predict(x_dev_arg), axis=1)
# print(classification_report(dev_labels, predictions))

# print('RNN, test data')
# x_test_arg = sequence.pad_sequences(test_features, maxlen=maxlen)
# predictions = np.argmax(model.predict(x_test_arg), axis=1)
# print(classification_report(test_labels, predictions))


# print('RNN, bal dev data')
# x_dev_arg_bal = sequence.pad_sequences(bal_dev_features, maxlen=maxlen)
# predictions = np.argmax(model.predict(x_dev_arg_bal), axis=1)
# print(classification_report(bal_dev_labels, predictions))

# print('RNN, bal test data')
# x_test_arg_bal = sequence.pad_sequences(bal_test_features, maxlen=maxlen)
# predictions = np.argmax(model.predict(x_test_arg_bal), axis=1)
# print(classification_report(bal_test_labels, predictions))


# Start of Seperate Editorial and Letters Experiments
print('RNN, editorial dev data')
x_dev_arg = sequence.pad_sequences(dev_features_edi, maxlen=maxlen)
predictions = np.argmax(model.predict(x_dev_arg), axis=1)
print(classification_report(dev_labels_edi, predictions))

print('RNN, editorial test data')
x_test_arg = sequence.pad_sequences(test_features_edi, maxlen=maxlen)
predictions = np.argmax(model.predict(x_test_arg), axis=1)
print(classification_report(test_labels_edi, predictions))


print('RNN, letters dev data')
x_dev_arg = sequence.pad_sequences(dev_features_let, maxlen=maxlen)
predictions = np.argmax(model.predict(x_dev_arg), axis=1)
print(classification_report(dev_labels_let, predictions))

print('RNN, letters test data')
x_test_arg = sequence.pad_sequences(test_features_let, maxlen=maxlen)
predictions = np.argmax(model.predict(x_test_arg), axis=1)
print(classification_report(test_labels_let, predictions))
# End of Seperate Editorial and Letters Experiments




# model 3: RNN + BERT
max_features = 3
maxlen = 100

input_emb = Input(shape=(768,))
# dense_1 = Dense(256, activation='relu', activity_regularizer=l2(0.0001))(input_emb)
# dropout_1 = Dropout(0.5)(dense_1)
dense_2 = Dense(128, activation='relu', activity_regularizer=l2(0.0001))(input_emb)
dropout_2 = Dropout(0.5)(dense_2)

input_arg = Input(shape=(maxlen,))
model_arg = Embedding(max_features, 128)(input_arg)
model_arg = SimpleRNN(128, dropout=0.2, recurrent_dropout=0.2)(model_arg)

merged = concatenate([dropout_2, model_arg])
dense_pred = (Dense(2, activity_regularizer=l2(0.0001), activation='softmax'))(merged)

model = Model(inputs=[input_emb, input_arg], outputs=dense_pred)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())



x_train_arg = sequence.pad_sequences(train_features, maxlen=maxlen)
x_dev_arg = sequence.pad_sequences(dev_features, maxlen=maxlen)

model.fit([train_embeddings_features, x_train_arg], train_labels_categorical,
          batch_size=batch_size,
          epochs=5,
          validation_data=([dev_embeddings_features, x_dev_arg], dev_labels_categorical))



# print('BERT + RNN, dev data')
# predictions = np.argmax(model.predict([dev_embeddings_features, x_dev_arg]), axis=1)
# print(classification_report(dev_labels, predictions))

# print('BERT + RNN, test data')
# x_test_arg = sequence.pad_sequences(test_features, maxlen=maxlen)
# score, acc = model.evaluate([test_embeddings_features, x_test_arg], test_labels_categorical)
# print(score, acc)
# predictions = np.argmax(model.predict([test_embeddings_features, x_test_arg]), axis=1)
# print(classification_report(test_labels, predictions))


# print('BERT + RNN, bal dev data')
# x_dev_arg_bal = sequence.pad_sequences(bal_dev_features, maxlen=maxlen)
# predictions = np.argmax(model.predict([bal_dev_emb, x_dev_arg_bal]), axis=1)
# print(classification_report(bal_dev_labels, predictions))

# print('BERT + RNN, bal test data')
# x_test_arg_bal = sequence.pad_sequences(bal_test_features, maxlen=maxlen)
# predictions = np.argmax(model.predict([bal_test_emb, x_test_arg_bal]), axis=1)
# print(classification_report(bal_test_labels, predictions))



# Start of Seperate Editorial and Letters Experiments
print('BERT + RNN, dev data')
x_dev_arg = sequence.pad_sequences(dev_features_edi, maxlen=maxlen)
predictions = np.argmax(model.predict([dev_emb_features_edi, x_dev_arg]), axis=1)
print(classification_report(dev_labels_edi, predictions))

print('BERT + RNN, test data')
x_test_arg = sequence.pad_sequences(test_features_edi, maxlen=maxlen)
predictions = np.argmax(model.predict([test_emb_features_edi, x_test_arg]), axis=1)
print(classification_report(test_labels_edi, predictions))


print('BERT + RNN, bal dev data')
x_dev_arg = sequence.pad_sequences(dev_features_let, maxlen=maxlen)
predictions = np.argmax(model.predict([dev_emb_features_let, x_dev_arg]), axis=1)
print(classification_report(dev_labels_let, predictions))

print('BERT + RNN, bal test data')
x_test_arg = sequence.pad_sequences(test_features_let, maxlen=maxlen)
predictions = np.argmax(model.predict([test_emb_features_let, x_test_arg]), axis=1)
print(classification_report(test_labels_let, predictions))
#End of Seperate Editorial and Letters Experiments





#### using logits
def artcile_split_and_pred_logits(sentences, predictions):
    'Retruns article splitted along with their sentences labels'
    
    sentences = sentences[1:]
    assert len(sentences) == len(predictions)
    
    article, article_splits = [], []
    for i, (sent, pred) in enumerate(zip(sentences, predictions)):
        if sent == 'ARTICLE_SPLIT_LINE\t0\n':
            article_splits.append(article)
            article = []
        else:
            article.append(( sent.split('\t')[0], pred))
    
    return article_splits

def prepare_data_logits(articles, sentences, predictions , mode='not_categorical'):
    
    arg = artcile_split_and_pred_logits(sentences, predictions)
    
    features, labels = [], []
    for article_arg, article_lable in zip(arg, articles):
        features.append(article_sent_labels_emb(article_arg))
        labels.append(int(article_lable.split('\t')[1].rstrip()))
    
    if mode == 'categorical':
        return np.array(features), np_utils.to_categorical(labels)
    else: # mode == 'not_categorial'
        return np.array(features), np.array(labels)

   
train_features, _ = prepare_data_logits(train_articles, train_sent, train_pred_logits)
dev_features, _ = prepare_data_logits(dev_articles, dev_sent, dev_pred_logits)
test_features, _ = prepare_data_logits(test_articles, test_sent, test_pred_logits)

# _, train_labels = prepare_data_logits(train_articles, train_sent, train_pred_logits, mode='categorical')
# _, dev_labels = prepare_data_logits(dev_articles, dev_sent, dev_pred_logits, mode='categorical')
# _, test_labels = prepare_data_logits(test_articles, test_sent, test_pred_logits, mode='categorical')



#RNN + BERT with prediction logits
max_features = 3
maxlen = 100

input_emb = Input(shape=(768,))
dense_1 = Dense(256, activation='relu', activity_regularizer=l2(0.0001))(input_emb)
dropout_1 = Dropout(0.5)(dense_1)
dense_2 = Dense(128, activation='relu', activity_regularizer=l2(0.0001))(dropout_1)
dropout_2 = Dropout(0.5)(dense_2)

input_arg = Input(shape=(maxlen,))
model_arg = Embedding(max_features, 128)(input_arg)
model_arg = SimpleRNN(128, dropout=0.2, recurrent_dropout=0.2)(model_arg)

merged = concatenate([dropout_2, model_arg])
dense_pred = (Dense(2, activity_regularizer=l2(0.0001), activation='softmax'))(merged)

model = Model(inputs=[input_emb, input_arg], outputs=dense_pred)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())



x_train_arg = sequence.pad_sequences(train_features, maxlen=maxlen)
x_dev_arg = sequence.pad_sequences(dev_features, maxlen=maxlen)

model.fit([train_embeddings_features, x_train_arg], train_labels_categorical,
          batch_size=batch_size,
          epochs=1,
          validation_data=([dev_embeddings_features, x_dev_arg], dev_labels_categorical))



print('BERT + RNN, dev data')
x_dev_arg = sequence.pad_sequences(dev_features, maxlen=maxlen)
predictions = np.argmax(model.predict([dev_embeddings_features, x_dev_arg]), axis=1)
print(classification_report(dev_labels, predictions))

print('BERT + RNN, test data')
x_test_arg = sequence.pad_sequences(test_features, maxlen=maxlen)
score, acc = model.evaluate([test_embeddings_features, x_test_arg], test_labels_categorical)
print(score, acc)
predictions = np.argmax(model.predict([test_embeddings_features, x_test_arg]), axis=1)
print(classification_report(test_labels, predictions))
