
'''
Example script to train a model to generate text from a text file.
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import Adam
import numpy as np
import random
import sys
import os
import codecs
import collections
from six.moves import cPickle


data_dir = 'data/Artistes_et_Phalanges-David_Campion'# data directory containing input.txt
save_dir = 'save' # directory to store models
rnn_size = 128 # size of RNN
batch_size = 30 # minibatch size
seq_length = 15 # sequence length
num_epochs = 8 # number of epochs
learning_rate = 0.001 #learning rate
sequences_step = 1 #step to create sequences


input_file = os.path.join(data_dir, "input.txt")
vocab_file = os.path.join(save_dir, "words_vocab.pkl")

#read data
with codecs.open(input_file, "r", encoding=None) as f:
    data = f.read()

x_text = data.split()

# count the number of words
word_counts = collections.Counter(x_text)

# Mapping from index to word : that's the vocabulary
vocabulary_inv = [x[0] for x in word_counts.most_common()]
vocabulary_inv = list(sorted(vocabulary_inv))

# Mapping from word to index
vocab = {x: i for i, x in enumerate(vocabulary_inv)}
words = [x[0] for x in word_counts.most_common()]

#size of the vocabulary
vocab_size = len(words)

#save the words and vocabulary
with open(os.path.join(vocab_file), 'wb') as f:
    cPickle.dump((words, vocab, vocabulary_inv), f)

#create sequences
sequences = []
next_words = []
for i in range(0, len(x_text) - seq_length, sequences_step):
    sequences.append(x_text[i: i + seq_length])
    next_words.append(x_text[i + seq_length])

print('nb sequences:', len(sequences))


print('Vectorization.')
X = np.zeros((len(sequences), seq_length, vocab_size), dtype=np.bool)
y = np.zeros((len(sequences), vocab_size), dtype=np.bool)
for i, sentence in enumerate(sequences):
    for t, word in enumerate(sentence):
        X[i, t, vocab[word]] = 1
    y[i, vocab[next_words[i]]] = 1


# build the model: a single LSTM
print('Build LSTM model.')
model = Sequential()
model.add(LSTM(rnn_size, input_shape=(seq_length, vocab_size)))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))

#adam optimizer
optimizer = Adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

#fit the model
model.fit(X, y,batch_size=batch_size,epochs=num_epochs)

#save the model
model.save(save_dir + "/" + 'my_model.h5')
