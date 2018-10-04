# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 15:46:35 2018

@author: C63165
"""

from Clusterer import importData,tokenize_and_stem
from nltk.tokenize import sent_tokenize
import random
from gensim.models import Doc2Vec
import gensim
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
#import matplotlib.pyplot as plt
import time
from sklearn.metrics.pairwise import cosine_similarity

#print(0)
def importNTokenize(filename):
    k=importData(filename)[1]
    k=[tokenize_and_stem(s) for s in k]
    k=[s for s in k if len(s)>5]
    print(len(k))
    return k

def tokenize(text):
    return tokenize_and_stem(text)

#print(1)
def labelSent(content):
    LabeledSentence1 = gensim.models.doc2vec.TaggedDocument
    j=0
    
    tagged_content=[]
#    print(type(content))
    for t in content:      
#        print(type(t))
        for s in sent_tokenize(t):
            tagged_content.append(LabeledSentence1(tokenize_and_stem(s),[j]))
        j+=1
    return tagged_content

def randColor():
    return'#'+random.choice('01234556789ABCDEF')+random.choice('01234556789ABCDEF')+random.choice('01234556789ABCDEF')+random.choice('01234556789ABCDEF')+random.choice('01234556789ABCDEF')+random.choice('01234556789ABCDEF')
#print(2)    
def d2v(tagged):
    d2v_model = Doc2Vec(tagged, size = 5000, window = 3, min_count = 10, workers=8, dm = 1, alpha=0.025, min_alpha=0.001)
    x=d2v_model.train(tagged, total_examples=d2v_model.corpus_count, epochs=10, start_alpha=0.002, end_alpha=-0.016)
    return d2v_model

def kMeans(model,n_clust):
    kmeans_model = KMeans(n_clusters=n_clust, init='k-means++', max_iter=100,n_init=100)  
    X = kmeans_model.fit(model.docvecs.doctag_syn0)
    labels=kmeans_model.labels_.tolist()
    return (kmeans_model,X,labels,n_clust)
#print(4)

#t0=time.time()
#k=importNTokenize('processed.json')
#tagged_content=labelSent(k)
#model=d2v(tagged_content)
#km=kMeans(model,100)
##plotCluster(km[0],model,km[2],100)
#t1=time.time()
#print(t1-t0)