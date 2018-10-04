# -*- coding: utf-8 -*-
"""
Created on Fri May  4 17:19:53 2018

@author: C63165
"""

from Indexer import Documents
from Indexer import getIntro
from Clusterer import *
import pickle
import json

def newCluster(model,processed):
    print(1)
    file=open(model,'rb')
    model=pickle.load(file)
    print(2)
    file.close()
    vect=model[0]
#    tfidf_matrix=model[1]
#    terms=model[2]
    km=model[3]
#    clusters=model[4]
    cdict=model[5]
    print(3)
    doclist=importData(processed)[0]
    print(4)
    file=open('processed.json','r',encoding='utf8')
    ddict=file.read()
    file.close()
    ddict=json.loads(ddict)
    cluster=Cluster(doclist,cdict,ddict,km,vect)
    return cluster

def loadCluster(doclist,cdict,ddict,km,vect):
    file=open(doclist,'rb')
    doclist=pickle.load(file)
    file.close()
    
    file=open(cdict,'rb')
    cdict=pickle.load(file)
    file.close()
    
    file=open(ddict,'rb')
    ddict=pickle.load(file)
    file.close()
    
    file=open(km,'rb')
    km=pickle.load(file)
    file.close()
    
    file=open(vect,'rb')
    vect=pickle.load(file)
    file.close()
    
    cluster=Cluster(doclist,cdict,ddict,km,vect)
    return cluster

def saveCluster(cluster,name):
    file=open(name,'wb')
    pickle.dump(cluster,file)
    file.close()
  
