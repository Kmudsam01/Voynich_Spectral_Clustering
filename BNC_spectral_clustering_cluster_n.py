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

import nltk.corpus.reader.bnc as bnc

#Time to do a similar analysis of an english text to see how the clusters relate

files = ["A00.xml", "A01.xml", "A02.xml"]
BNC = bnc.BNCCorpusReader(root = '/Users/karlmudsam/BNC/Texts/A/A0/', fileids = files)
BNC_sentences = BNC.sents()
BNC_tokens = BNC.words()

print(len(BNC_tokens))

remove_punctuation = lambda x: "".join([char for char in list(x) if char.isalpha()]) #gets rid of punctuation by splitting up a string, and only picking out the alphanumeric characters, and then joining them

BNC_vocab = list(set(BNC_tokens))

#creating the word2vec model
model = gensim.models.Word2Vec(BNC_sentences, min_count = 1, vector_size = 100, window = 5)


#initializing similarity matrix
similarity_matrix = np.matrix(np.zeros( (len(BNC_vocab), len(BNC_vocab))) )

#populating similarity matrix, excluding diagonal
for i in range(len(BNC_vocab)):
    for j in range(len(BNC_vocab)):
        print(i,j, "<- indexes")
        print(model.wv.similarity(BNC_vocab[i], BNC_vocab[j]))
        if i == j:
            similarity_matrix[i,j] = 0
        else:
            similarity_matrix[i,j] = model.wv.similarity(BNC_vocab[i], BNC_vocab[j])

#intializing degree matrix
degree_matrix = np.matrix(np.zeros( (len(BNC_vocab), len(BNC_vocab))) )

column_sums = similarity_matrix.sum(axis = 0)

#populating degree matrix
for j in range(len(BNC_vocab)):
    degree_matrix[j,j] = column_sums[0,j]


#Creating the normalized laplacian

#setup, finding all the terms that go into the normalized laplacian
degree_matrix_inverse = np.linalg.inv(degree_matrix)
degree_matrix_inv_sqrt = scipy.linalg.sqrtm(degree_matrix_inverse)
D = np.real(degree_matrix_inv_sqrt)
A = similarity_matrix
I = np.identity(len(BNC_vocab))

#actually doing the computation to find the normalized laplacian
L = I - (D@(A@D))


#finding the eigenvalues
eig = np.linalg.eig(L)

#sorting 'em
zipped_vals_vects = zip(eig[0], eig[1])
sorted_vals_vects = sorted(zipped_vals_vects, key = lambda x: x[0])

#Note: order eigenvectors
eigenvector_matrix = np.zeros(shape = (len(BNC_vocab), len(BNC_vocab)))

for i in range(len(sorted_vals_vects)):
    eigenvector_matrix[i] = sorted_vals_vects[i][1]

eigenvector_matrix = np.matrix(eigenvector_matrix)

eigenvector_matrix = eigenvector_matrix.T

truncated_eigenvector_matrix = eigenvector_matrix[:,0:20]

truncated_eigenvector_matrix = truncated_eigenvector_matrix

#finding optimal number of clusters
#what we do here is that we run kmeans 50 times and record the score for each set of labels it produces, then take the average
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
plt.title("Silhouette Score vs Cluster Count in the \nBritish National Corpus files A00, A02, A03")
plt.xlabel("Cluster Count (n)")
plt.ylabel("Silhouette Score")
plt.ylim(ymin = 0, ymax = 0.6)
plt.show()

