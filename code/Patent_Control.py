# -*- coding: utf-8 -*-
"""
Created on Sun May 19 12:04:28 2019

@author: smith
"""





import numpy as np
import pandas as pd
from datetime import datetime
import os



# Code directory
os.chdir('C:\\Users\\smith\\Patent_Project\\code')
from patent_class import extraction



# =============================================================================
# Load Patent Data
# =============================================================================

# Data directory
data_directory = 'C:\\Users\\smith\\Patent_Project'


# Load patents
start = datetime.now()
patents = extraction.extract_patent_files(data_directory + '\\txtfields')
print('Time to extract patent data:', datetime.now() - start)
print('Dimensions of patent dataset:', patents.shape)


# Load patent labels
patent_labels = pd.read_csv(data_directory + '\\pubtri\\pubtri.csv', 
                                     dtype='object', 
                                     header=None, 
                                     names=['pub', 'label'])

print('Dimension of labels dataset:', patent_labels.shape)
print('Number of unique labels:', len(patent_labels.label.unique()) )


# subset to only take classifications that have at least 500 instances
thresh = 500
label_counts = patent_labels.label.value_counts()
selected_labels = label_counts[label_counts >= thresh ]
patent_labels = patent_labels.loc[ patent_labels.label.isin(selected_labels.index.values )]
print('Shape of labels dataset after removal of low count labels:', patent_labels.shape)

# keep only the first character of the classifications to reduce the complexity
patent_labels.label = patent_labels.label.apply(lambda x: x[0])

# Append labels to the patent dataset
patents = pd.merge(patents, patent_labels, how='left', left_on='pub', right_on='pub')
print('Dimensions of patent dataset after merging on labels:', patents.shape )

# Remove rows with no label
patents.dropna(axis=0, subset=['label'], inplace=True)
print('Dimensions of patent data after dropping nan:', patents.shape )
print('Number of unique labels on final dataset:', len(patents.label.unique()))

# Check the final distribution
patents.label.value_counts() / patents.shape[0]


# =============================================================================
# Preprocessing
# =============================================================================


# preprocess the labels
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

label_enc = LabelEncoder()

labels = label_enc.fit_transform(patents.label.values ) # labels are now numeric
print('Length of labels vector:', len(labels))
print('Number of classes:', len(label_enc.classes_) )

np.save('C:\\Users\\smith\\Patent_Project\\data\\classes.npy', label_enc.classes_)


# investigation of using abstract or claims
#len_abstract = patents['abstract'].apply(lambda x: len(x))
#len_claims = patents['claims'].apply(lambda x: len(x))
#diff_len = len_claims - len_abstract
#diff_len.idxmin()
#len_abstract[ diff_len < 0 ]
#
#patents.loc[ len_abstract[ len_abstract < 50].index , 'abstract'].shape
#patents.loc[ len_claims[ len_claims < 100].index , 'claims'].shape

# should we remove rows where even the abstract isn't long enough?

# use abstract if it is longer than claim
texts = patents[['abstract', 'claims']].apply(lambda row: row['abstract'] if 
               len(row['abstract']) > len(row['claims']) 
               else row['claims'], axis=1)

maxlen = max([len(s.split()) for s in texts])
print('Maximum length of texts:', maxlen)

del patents, patent_labels, selected_labels  # remove patent data from memory



from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

max_words = 80000   # considers only the top 80,000 words in the dataset

# apply tokenizer to texts
tokenizer = Tokenizer(num_words = max_words) #num_words = max_words
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# pad the sequences so all same length
data = pad_sequences(sequences, maxlen = 5000)


print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

del sequences, texts




# =============================================================================
# Train/Val Split
# =============================================================================

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

# saving for reproducibility
np.save('C:\\Users\\smith\\Patent_Project\\data\\indices.npy', indices, allow_pickle=True)

training_samples = 180000   
validation_samples = 20000

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

print('Dimensions of training data (x_train, y_train):', x_train.shape, y_train.shape)
print('Dimensions of validation data (x_val, y_val):', x_val.shape, y_val.shape)

# compute the class weights from the training split
class_weight_list = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
np.save('C:\\Users\\smith\\Patent_Project\\data\\class_weight_list.npy', class_weight_list)


# vectorize the y sequences (since multiclass)
def vectorize_sequences(seq, dimension):
    results = np.zeros((len(seq), dimension))
    for i,s in enumerate(seq):
        results[i,s] = 1.
    return results

y_train_vec = vectorize_sequences(y_train, len(label_enc.classes_))
print('Dimensions of (vectorized) y_train_vec:', y_train_vec.shape)

y_val_vec = vectorize_sequences(y_val, len(label_enc.classes_))
print('Dimensions of (vectorized) y_val_vec:', y_val_vec.shape)


# =============================================================================
# Load Word Embeddings
# =============================================================================

glove_dir = 'D:\\NLP_Research\\Embeddings\\glove.6B'

embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.300d.txt'), encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_dim = 300 # since we extract glove.6B.300d.txt

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector  # words not found in the embedding index will be 0


print('Dimensions of Embedding Matrix:', embedding_matrix.shape)



# =============================================================================
# Modeling
# =============================================================================

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, LSTM, GRU, SpatialDropout1D
from keras.initializers import Constant
from datetime import datetime



# model architecture taken from:
# https://towardsdatascience.com/machine-learning-word-embedding-sentiment-classification-using-keras-b83c28087456
# https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17
model = Sequential()
model.add(Embedding(max_words, 
                    embedding_dim,
                    embeddings_initializer = Constant(embedding_matrix),
                    input_length = 5000,
                    trainable = False))
#model.add(SpatialDropout1D(0.2))
#model.add(LSTM(200, dropout=0.2, recurrent_dropout=0.2))
model.add(GRU(units=100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(len(label_enc.classes_), activation='softmax'))
model.summary()



start = datetime.now()

model.compile(optimizer = 'rmsprop',
             loss='categorical_crossentropy',
              metrics=['acc'])

history = model.fit(x_train, y_train_vec,
                   epochs=3, 
                   batch_size=60, 
                   class_weight = class_weight_list,
                   validation_data=(x_val, y_val_vec))

model.save_weights('C:\\Users\\smith\\Patent_Project\\models\\model_w_pretrain_GloVe_1char.h5')

print('Time to execute ', datetime.now() - start)

