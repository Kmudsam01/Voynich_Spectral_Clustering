#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 19:52:38 2024

@author: karlmudsam
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 18:22:54 2024

@author: karlmudsam
"""

import numpy as np
import gensim
from gensim.models import Word2Vec
import scipy

import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

from random import choice


vocab = []
num_unique_tokens = 5000
for i in range(num_unique_tokens):
    word_length = np.random.geometric(1/5)
    word = ""
    for j in range(word_length):
        word = word + chr(np.random.random_integers(97,122)) #this is a random letter a-z
    vocab.append(word)
    

used_vocab = set([])
sentences = []
num_sentences = 2000
for i in range(num_sentences):
    sentence_length = np.random.geometric(1/10)
    temp_sentence = []
    for j in range(sentence_length + 5):
        added_word = choice(vocab)
        temp_sentence.append(added_word)
        used_vocab.add(added_word)
    sentences.append(temp_sentence)

used_vocab = list(used_vocab)



#creating the word2vec model
model = gensim.models.Word2Vec(sentences, min_count = 1, vector_size = 100, window = 5)

#initializing similarity matrix
similarity_matrix = np.matrix(np.zeros( (len(used_vocab), len(used_vocab))) )

#populating similarity matrix, excluding diagonal
for i in range(len(used_vocab)):
    for j in range(len(used_vocab)):
        print(i,j, "<- indexes")
        print(model.wv.similarity(used_vocab[i], used_vocab[j]))
        if i == j:
            similarity_matrix[i,j] = 0
        else:
            similarity_matrix[i,j] = model.wv.similarity(used_vocab[i], used_vocab[j])

#intializing degree matrix
degree_matrix = np.matrix(np.zeros( (len(used_vocab), len(used_vocab)) ))

column_sums = similarity_matrix.sum(axis = 0)

#populating degree matrix
for j in range(len(used_vocab)):
    degree_matrix[j,j] = column_sums[0,j]


#Creating the normalized laplacian

#setup, finding all the terms that go into the normalized laplacian
degree_matrix_inverse = np.linalg.inv(degree_matrix)
degree_matrix_inv_sqrt = scipy.linalg.sqrtm(degree_matrix_inverse)
D = np.real(degree_matrix_inv_sqrt)
A = similarity_matrix
I = np.identity(len(used_vocab))

#actually doing the computation to find the normalized laplacian
L = I - (D@(A@D))


#finding the eigenvalues
eig = np.linalg.eig(L)

#sorting 'em
zipped_vals_vects = zip(eig[0], eig[1])
sorted_vals_vects = sorted(zipped_vals_vects, key = lambda x: x[0])

#Note: order eigenvectors
eigenvector_matrix = np.zeros(shape = (len(used_vocab), len(used_vocab)))

for i in range(len(sorted_vals_vects)):
    eigenvector_matrix[i] = sorted_vals_vects[i][1]

eigenvector_matrix = np.matrix(eigenvector_matrix)

eigenvector_matrix = eigenvector_matrix.T

truncated_eigenvector_matrix = eigenvector_matrix[:,0:20]

truncated_eigenvector_matrix = truncated_eigenvector_matrix

#finding optimal number of clusters

scores = []
max_clusters = 300
n = 50
for i in range(max_clusters - 2):
    print("Number of Clusters: ", i)
    kmeans = KMeans(n_clusters=i+2, random_state = 303).fit(truncated_eigenvector_matrix)
    labels = kmeans.labels_
        
    score = silhouette_score(truncated_eigenvector_matrix, labels)
    scores.append(score)

clusters = np.arange(2,max_clusters)
plt.plot(clusters, scores)
plt.title("Silhouette Score vs Cluster Count in the \nRandom Language")
plt.xlabel("Cluster Count (n)")
plt.ylabel("Silhouette Score")
plt.ylim(ymin = 0, ymax = 0.6)
plt.show()

