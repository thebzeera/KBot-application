# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 15:17:26 2018

@author: C63165
"""

# from Indexer import processJSON
from Indexer import Document, processJSON, Documents, getIntro
import json
import numpy as np
import os
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

stopwords = nltk.corpus.stopwords.words('english')
from nltk.stem.snowball import SnowballStemmer
import re
stemmer = SnowballStemmer("english")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans

direc = os.getcwd()
# from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
lm = WordNetLemmatizer()
# from Doc2vec_search import eDist,predict,getCentroids1
from scipy.sparse import csr_matrix
import numpy as np


def importData(filename):
    try:
        f = open(filename, 'r', encoding='utf8')
        ddict = json.loads(f.read())

        f.close()
    except:
        ddict = processJSON(filename)
    doclist = []
    for d in ddict:
        try:
            doclist.append(Document(ddict[d]))
        except:
            continue

    strlist = [d.string for d in doclist]
    print(len(strlist))
    return (doclist, strlist)


def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []

    for token in tokens:
        # if token.isalpha():
        filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens if t not in stopwords and len(t) > 2]
    return stems


def buildVocabFrame(contents):
    vocabulary = [w.lower() for s in contents for w in word_tokenize(s) if w not in stopwords]
    vocabulary = list(set(vocabulary))
    vocab_stemmed = [stemmer.stem(w) for w in vocabulary]
    vocab_frame = pd.DataFrame({'words': vocabulary}, index=vocab_stemmed)
    print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
    return vocab_frame


def vectorize(contents):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000, min_df=0, stop_words='english', use_idf=True,
                                       tokenizer=tokenize_and_stem, ngram_range=(1, 3))
    tfidf_matrix = tfidf_vectorizer.fit_transform(contents)
    terms = tfidf_vectorizer.get_feature_names()
    #    dist = 1 - cosine_similarity(tfidf_matrix)
    return (tfidf_vectorizer, tfidf_matrix, terms)


def trainKMeans(matrix, num_clusters):
    km = KMeans(n_clusters=num_clusters, n_init=50, init='k-means++', max_iter=100)
    km.fit(matrix)
    clusters = km.labels_.tolist()
    return (km, clusters)


def train(filename, num_clusters):
    contents = importData(filename)
    contents = contents[1]
    vector = vectorize(contents)
    vectorizer = vector[0]
    tfidf_matrix = vector[1]
    terms = vector[2]
    kmm = trainKMeans(tfidf_matrix, num_clusters)
    km = kmm[0]
    clusters = kmm[1]
    cdict = {}
    for c in range(len(clusters)):
        if clusters[c] in cdict:
            cdict[clusters[c]].append(c)
        else:
            cdict[clusters[c]] = []
            cdict[clusters[c]].append(c)

    return (vectorizer, tfidf_matrix, terms, km, clusters, cdict)


class Cluster(object):
    def __init__(self, doclist, cdict, ddict, vect, matrix):
        self.doc_clust = {}
        self.docs = Documents('the_daddy')
        for c in cdict:
            docs = Documents(str(c))
            for d in cdict[c]:
                doc = doclist[d]
                docs.addDoc(doc)
                self.docs.addDoc(doc)
            self.doc_clust[c] = docs
        self.docs.update()
        for d in self.doc_clust:
            self.doc_clust[d].IDF = self.docs.IDF
            self.doc_clust[d].computeTFIDF()
        self.doc_clust[len(self.doc_clust)] = self.docs
        self.ddict = ddict
        self.vect = vect
        stopfile = open('stopwords.txt', 'r', encoding='utf8')
        stopwords = stopfile.read()
        self.stopwords = stopwords.split()
        stopfile.close()
        self.matrix = matrix
        self.centroids = getCentroids(cdict, matrix, len(cdict)).toarray()

    def clustSearch(self, query, dist):
        v = self.vect.transform([query]).toarray()
        c = predict(self.centroids, v, dist)
        docs = self.doc_clust[c]
        query=re.split(r' ', query)
        query = [q for q in query if q not in self.stopwords]
        result = []
        test = [(ps.stem(lm.lemmatize(q)) in docs.wordlist) for q in query]
        if False not in test:
            result = docs.getMatch(query, 3)
        if not result:
            print('Miss!')
            result = self.docs.getMatch(query, 3)
        return result

    def clustprediction(self, query, dist, c):
        v = self.vect.transform([query]).toarray()
        docs = self.doc_clust[c]
        query = re.split(r' ', query)
        query = [q for q in query if q not in self.stopwords]
        print(query)
        result = []
        test = [(ps.stem(lm.lemmatize(q)) in docs.wordlist) for q in query]
        if False not in test:
            result = docs.getMatch(query, 3)
        if not result:
            print('Miss!')
            result = self.docs.getMatch(query, 3)
        return result

    def clustrec(self, query, dist):
        v = self.vect.transform([query]).toarray()
        c = predict(self.centroids, v, dist)
        return c

    def getParaList(self, ID):
        para = []
        for d in self.ddict:
            if self.ddict[d]["_id"]["$oid"] == ID:
                para = self.ddict[d]['content']['contents']
                break
        return para


def eDist(v1, v2):
    return np.sqrt(np.sum(np.square(v1 - v2), axis=1))


def cDist(v1, v2):
    v = np.dot(v2, v1.T)
    for i in range(len(v[0])):
        v[0][i] = v[0][i] / (np.linalg.norm(v2) * np.linalg.norm(v1[i]))
    return 1 - v


def predict(M, v, dist='c'):
    if dist == 'e':
        return np.argmin(eDist(M, v))
    elif dist == 'c':
        return np.argmin(cDist(M, v))


def getCentroids(cdict, matrix, n_clust):
    centroids = csr_matrix((n_clust, matrix.shape[1]))
    for c in cdict:
        cent = 0
        n = len(cdict[c])
        for d in cdict[c]:
            cent += matrix[d]
        cent /= n
        centroids[c] = cent
    return centroids



