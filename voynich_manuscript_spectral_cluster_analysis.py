#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 13:23:34 2023

@author: karlmudsam
"""

import numpy as np
import gensim
from gensim.models import Word2Vec
import scipy

path = """/Users/karlmudsam/Desktop/Voynich_Spectral_Clustering/Voynich_A_Maximal_Text.txt"""

voynich_text_file = open(path)
voynich_text = voynich_text_file.read()
voynich_tokenized = voynich_text.split()
voynich_tokenized = [i for i in voynich_tokenized if i.isalpha()]

print(len(voynich_tokenized))

voynich_sentences = []

#Turn tokenization into sentences, because that's what gensim's word2vec model wants
i = 0
while i < len(voynich_tokenized):
    sentence = voynich_tokenized[i:i+5]
    voynich_sentences.append(sentence)
    i = i + 5

#getting all the unique tokens
voynich_vocab = list(set(voynich_tokenized))

#creating the word2vec model
model = gensim.models.Word2Vec(voynich_sentences, min_count = 1, vector_size = 100, window = 5)

#initializing similarity matrix
similarity_matrix = np.matrix(np.zeros( (len(voynich_vocab), len(voynich_vocab))) )

#populating similarity matrix, excluding diagonal
for i in range(len(voynich_vocab)):
    for j in range(len(voynich_vocab)):
        print(i,j, "<- indexes")
        print(model.wv.similarity(voynich_vocab[i], voynich_vocab[j]))
        if i == j:
            similarity_matrix[i,j] = 0
        else:
            similarity_matrix[i,j] = model.wv.similarity(voynich_vocab[i], voynich_vocab[j])

#intializing degree matrix
degree_matrix = np.matrix(np.zeros( (len(voynich_vocab), len(voynich_vocab))) )

column_sums = similarity_matrix.sum(axis = 0)

#populating degree matrix
for j in range(len(voynich_vocab)):
    degree_matrix[j,j] = column_sums[0,j]


#Creating the normalized laplacian

#setup, finding all the terms that go into the normalized laplacian
degree_matrix_inverse = np.linalg.inv(degree_matrix)
degree_matrix_inv_sqrt = scipy.linalg.sqrtm(degree_matrix_inverse)
D = np.real(degree_matrix_inv_sqrt)
A = similarity_matrix
I = np.identity(len(voynich_vocab))

#actually doing the computation to find the normalized laplacian
L = I - (D@(A@D))


#finding the eigenvalues
eig = np.linalg.eig(L)

#sorting 'em
zipped_vals_vects = zip(eig[0], eig[1])
sorted_vals_vects = sorted(zipped_vals_vects, key = lambda x: x[0])


#When this finally finishes runnning, we'll do the following:
    #Create a matrix from k eigenvectors, where each row corresponds to a node
    #Cluster these row vectors using kmeans
    #Find optimal number of clusters using elbow method

#Also extra note: do this with a comparably sized english corpus of a similar topic (i.e. plants and such, maybe from approx same time period)

#thank god it took 30m to run 

#Note: order eigenvectors
eigenvector_matrix = np.zeros(shape = (len(voynich_vocab), len(voynich_vocab)))

for i in range(len(sorted_vals_vects)):
    eigenvector_matrix[i] = sorted_vals_vects[i][1]

eigenvector_matrix = np.matrix(eigenvector_matrix)

eigenvector_matrix = eigenvector_matrix.T

truncated_eigenvector_matrix = eigenvector_matrix[:,0:20]

truncated_eigenvector_matrix = truncated_eigenvector_matrix


#time to cluster!!

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


#How many clusters??? Silhouette score! We'll import it from sklearn

from sklearn.metrics import silhouette_score


scores = []
max_clusters = 300
n = 50
for i in range(max_clusters - 1):
    print("Number of Clusters: ", i)
    kmeans = KMeans(n_clusters=i+2, random_state = 303).fit(truncated_eigenvector_matrix)
    labels = kmeans.labels_
        
    score = silhouette_score(truncated_eigenvector_matrix, labels)
    scores.append(score)

clusters = np.arange(2,max_clusters+1)
plt.plot(clusters, scores)
plt.title("Silhouette Score vs Cluster Count in the \nComplete Voynich Manuscript")
plt.xlabel("Cluster Count (n)")
plt.ylabel("Silhouette Score")
plt.ylim(ymin = 0, ymax = 0.6)
plt.show()

'''
disct_word_and_label = dict(word_and_label)

#Time to plot some stuff!
import networkx as nx
import itertools as it

index = group_2_indexes + group_3_indexes + group_4_indexes

selected_graph = np.zeros(shape = (len(index), len(index)))

for i,j in it.combinations(index, 2):
    if i != j:
        x = index.index(i)
        y = index.index(j)
        selected_graph[x,y] = similarity_matrix[i,j]
        selected_graph[y,x] = similarity_matrix[j,i]
    else:
        selected_graph[i,j] = 0
        selected_graph[j,i] = 0
    




rows, cols = np.where(selected_graph != 0)
edges = zip(rows.tolist(), cols.tolist())
gr = nx.Graph()
gr.add_edges_from(edges, length = 30)

for i in range(len(selected_graph)):
    for j in range(len(selected_graph)):
        if i != j:
            gr.edges[i,j]["weight"] = 1 / selected_graph[i,j]          
            
color_map = []
for i in range(len(selected_graph)):
    voynich_vocab_index = index[i]
    token = voynich_vocab[voynich_vocab_index]
    if word_and_label_dict[token] == 0:
        color_map.append("blue")
        
    if word_and_label_dict[token] == 1:
        color_map.append("red")
        
    if word_and_label_dict[token] == 2:
        color_map.append("yellow")
        
    if word_and_label_dict[token] == 3:
        color_map.append("orange")

label_dict = {}

for i in range(len(selected_graph)):
    voynich_vocab_index = index[i]
    token = voynich_vocab[voynich_vocab_index]
    label_dict[i] = token
        
pos = nx.layout.spring_layout(gr)
nx.draw_networkx_edges(gr, pos, alpha = 0.05)
nx.draw_networkx_nodes(gr, pos, node_size=20, node_color=color_map)#draw nodes
nx.draw_networkx_labels(gr, pos, label_dict)

plt.show()


#Finally we're going to give some descriptive statistics on our silhouette score for the clustering

sil_scores = []

for i in range(100):
    kmeans = KMeans(n_clusters=4).fit(truncated_eigenvector_matrix)
    labels = kmeans.labels_
    
    score = silhouette_score(truncated_eigenvector_matrix, labels)
    sil_scores.append(score)

print("Mean Silhouette Score: ", np.mean(sil_scores))

print("Variance of Silhouette Scores: ", np.std(sil_scores))
'''

    














    
    
    





