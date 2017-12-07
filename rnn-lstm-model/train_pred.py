#!/usr/bin/python

from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import dynamic_rnn as rnn
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from keras.datasets import imdb

from attention import attention
from utils import *

from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedShuffleSplit
from itertools import izip
from nltk.tokenize import TweetTokenizer
from sklearn.cluster import KMeans



train_data= "../train dataset/Tweet.csv"
train_topic= "../train dataset/Target.csv"
train_label= "../train dataset/Stance.csv"

test_data= "../test dataset/Tweet.csv"
test_topic= "../test dataset/Target.csv"
test_label= "../test dataset/Stance.csv"


target_dict={}
stance_dict={}
inv_target_dict={}
inv_stance_dict={}

x=set()
with open("../train dataset/Target.csv", "rb") as f:
    for row in f:
        x.add(row.strip())
x=list(x)
i=0
for tar in x:
    target_dict[tar]=i
    inv_target_dict[i]=tar
    i+=1

x=set()
with open("../train dataset/Stance.csv", "rb") as f:
    for row in f:
        x.add(row.strip())
x=list(x)
i=0
for tar in x:
    stance_dict[tar]=i
    inv_stance_dict[i]=tar  
    i+=1


# print target_dict,stance_dict 
tknzr=TweetTokenizer()
x_train, y_train = [[] for i in range(5)], [[] for i in range(5)]
X_train, Y_train = [[] for i in range(5)], [[] for i in range(5)]

with open("../train dataset/Tweet.csv", "rb") as f1, open("../train dataset/Target.csv", "rb") as f2, open("../train dataset/Stance.csv", "rb") as f3:
    for l1,l2,l3 in izip(f1,f2,f3):
        
        tweet=tknzr.tokenize(l1.strip())
        x_train[target_dict[l2.strip()]].append(tweet)
        y_train[target_dict[l2.strip()]].append(l3.strip())

x_dev, y_dev = [[] for i in range(5)], [[] for i in range(5)]
X_dev, Y_dev = [[] for i in range(5)], [[] for i in range(5)]


with open("../dev dataset/Tweet.csv", "rb") as f1, open("../dev dataset/Target.csv", "rb") as f2, open("../dev dataset/Stance.csv", "rb") as f3:
    for l1,l2,l3 in izip(f1,f2,f3):

        tweet=tknzr.tokenize(l1.strip())
        x_dev[target_dict[l2.strip()]].append(tweet)
        y_dev[target_dict[l2.strip()]].append(l3.strip())        

x_test, y_test = [[] for i in range(5)], [[] for i in range(5)]
X_test, Y_test = [[] for i in range(5)], [[] for i in range(5)]


with open("../test dataset/Tweet.csv", "rb") as f1, open("../test dataset/Target.csv", "rb") as f2, open("../test dataset/Stance.csv", "rb") as f3:
    for l1,l2,l3 in izip(f1,f2,f3):

        tweet=tknzr.tokenize(l1.strip())
        x_test[target_dict[l2.strip()]].append(tweet)
        y_test[target_dict[l2.strip()]].append(l3.strip())




all_words=[set(w for sen in x_train[i] for w in sen) for i in range(5)]
  
word_idx=[{} for i in range(5)]
for i in xrange(5):
    j=0;
    for word in all_words[i]:
        word_idx[i][word]=j
        j+=1

NUM_WORDS = 10000
INDEX_FROM = 3
SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 100
HIDDEN_SIZE = 150
ATTENTION_SIZE = 50
KEEP_PROB = 0.8
BATCH_SIZE = 20
NUM_EPOCHS = 10000
DELTA = 0.5
learning_rate=0.05


    

vocabulary_size=[None for _ in range(5)]
f=open("Prediction.csv","wb")
from random import shuffle

def classifier(X):
    pred=np.array([[x] for x in X])
    kmeans = KMeans(n_clusters=3, random_state=0).fit(pred)
    centres=np.sort(kmeans.cluster_centers_) 
    res=[]
    for elem in X:
        val=0
        dist=float("inf")
        for i in xrange(3):
            if(abs(elem-centres[i])<dist):
                dist=abs(elem-centres[i])
                val=i
        res.append(val)
    return np.array(res)

for i in xrange(5):
    x_train[i]=convert_into_idx(x_train[i], word_idx[i])
    vocabulary_size[i] = get_vocabulary_size(x_train[i])
    x_test[i] = fit_in_vocabulary(x_test[i],vocabulary_size[i], word_idx[i])
    x_dev[i] = fit_in_vocabulary(x_dev[i],vocabulary_size[i], word_idx[i])
    X_train[i] = zero_pad(x_train[i], SEQUENCE_LENGTH)
    X_dev[i] = zero_pad(x_dev[i], SEQUENCE_LENGTH)
    X_test[i] = zero_pad(x_test[i], SEQUENCE_LENGTH)
    Y_train[i] = encoding(y_train[i],stance_dict)
    Y_dev[i] = encoding(y_dev[i],stance_dict)
    Y_test[i] = encoding(y_test[i],stance_dict)
        
    batch_ph = tf.placeholder(tf.int32, [None, SEQUENCE_LENGTH])
    target_ph = tf.placeholder(tf.float32, [None])
    seq_len_ph = tf.placeholder(tf.int32, [None])
    keep_prob_ph = tf.placeholder(tf.float32)

    # Embedding layer
    embeddings_var = tf.Variable(tf.random_uniform([vocabulary_size[i], EMBEDDING_DIM], -1.0, 1.0), trainable=True)
    batch_embedded = tf.nn.embedding_lookup(embeddings_var, batch_ph)

    # (Bi-)RNN layer(-s)
    with tf.variable_scope(str(i)):
        rnn_outputs, _ = bi_rnn(GRUCell(HIDDEN_SIZE), GRUCell(HIDDEN_SIZE),
                            inputs=batch_embedded, sequence_length=seq_len_ph, dtype=tf.float32)
    
    # Attention layer
    attention_output, alphas = attention(rnn_outputs, ATTENTION_SIZE, return_alphas=True)
    
    drop = tf.nn.dropout(attention_output, keep_prob_ph)
    W = tf.Variable(tf.truncated_normal([drop.get_shape()[1].value, 1], stddev=0.1))
    b = tf.Variable(tf.constant(0., shape=[1]))
    y_hat = tf.nn.xw_plus_b(drop, W, b)

    y_hat = tf.squeeze(y_hat)

    # Cross-entropy loss and optimizer initialization
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=target_ph))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # Accuracy metric
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.sigmoid(y_hat)), target_ph), tf.float32))

    # Actual lengths of sequences
    seq_len_dev = np.array([list(x).index(0) + 1 for x in X_dev[i]])
    seq_len_test = np.array([list(x).index(0) + 1 for x in X_test[i]])
    seq_len_train = np.array([list(x).index(0) + 1 for x in X_train[i]])

    # Batch generators
    train_batch_generator = batch_generator(X_train[i], Y_train[i], BATCH_SIZE)
    test_batch_generator = batch_generator(X_test[i], Y_test[i], BATCH_SIZE)
    dev_batch_generator = batch_generator(X_dev[i], Y_dev[i], BATCH_SIZE)

    saver = tf.train.Saver()

    if __name__ == "__main__":
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print("Start learning...")
            for epoch in range(NUM_EPOCHS):
                loss_train = 0
                loss_test = 0
                accuracy_train = 0
                accuracy_test = 0

                print("epoch: {}\t".format(epoch), end="")

                # Training
                num_batches = X_train[i].shape[0] / BATCH_SIZE
                for b in range(num_batches):
                    x_batch, y_batch = train_batch_generator.next()
                    seq_len = np.array([list(x).index(0) + 1 for x in x_batch])  # actual lengths of sequences
                    temp=x_batch
                    loss_tr, acc, _ = sess.run([loss, accuracy, optimizer], feed_dict={batch_ph: x_batch, target_ph: y_batch, seq_len_ph: seq_len, keep_prob_ph: KEEP_PROB})
                    accuracy_train += acc
                    loss_train = loss_tr * DELTA + loss_train * (1 - DELTA)
                accuracy_train /= num_batches

                # Testing
                num_batches = X_dev[i].shape[0] / BATCH_SIZE
                for b in range(num_batches):
                    x_batch, y_batch = dev_batch_generator.next()
                    temp=x_batch
                    seq_len = np.array([list(x).index(0) + 1 for x in x_batch])  # actual lengths of sequences
                    y_hatv, loss_test_batch, acc = sess.run([y_hat, loss, accuracy], feed_dict={batch_ph: x_batch, target_ph: y_batch, seq_len_ph: seq_len, keep_prob_ph: 1.0})
                    accuracy_test += acc
                    loss_test += loss_test_batch
                accuracy_test /= num_batches
                loss_test /= num_batches
                
                print("loss: {:.3f}, val_loss: {:.3f}, acc: {:.3f}, val_acc: {:.3f}".format(
                    loss_train, loss_test, accuracy_train, accuracy_test
                ))
            c = list(zip(X_test[i],Y_test[i]))
            shuffle(c)
            X_test[i],Y_test[i]=zip(*c)
            for j in xrange(0, len(X_test[i]), 10):
                x_batch_test, y_batch_test = X_test[i][j:j+10], Y_test[i][j:j+10]
                seq_len_test = np.array([list(x).index(0) + 1 for x in x_batch_test])
                alphas_test, y_hatv = sess.run([alphas,y_hat], feed_dict={batch_ph: x_batch_test, target_ph: y_batch_test,seq_len_ph: seq_len_test, keep_prob_ph: 1.0})
                pred=classifier(y_hatv)
                for p,r in izip(pred,y_batch_test):
                    f.write(inv_stance_dict[p]+"\t"+inv_stance_dict[r]+"\n")

            saver.save(sess, "model_"+ str(i))
            
f.close()
