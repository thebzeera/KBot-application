# -*- coding: utf-8 -*-
"""
Created on Wed May  9 09:33:50 2018

@author: C63165
"""
from Clusterer import trainKMeans,tokenize_and_stem
import pickle
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
def loadModel(TFIDF,STRLIST,VOCAB):
    file=open(TFIDF,'rb')
    tfidf=pickle.load(file)
    file.close()
    
#    matrix=tfidf[1]
#    terms=tfidf[2]
#    vect=tfidf[0]
    

    
    file=open(STRLIST,'rb')
    strlist=pickle.load(file)
    file.close()
    
    file=open(VOCAB,'rb')
    vocab_frame=pickle.load(file)
    file.close()
    
    return tfidf,strlist,vocab_frame

def getClusterDict(km):
    clusters = km[0].labels_.tolist()
    
    cdict={}
    for c in range(len(clusters)):
        if clusters[c] in cdict:
            cdict[clusters[c]].append(c)
        else:
            cdict[clusters[c]]=[]
            cdict[clusters[c]].append(c) 
    return clusters,cdict
       
def getCentroids(cdict,matrix,n_clust)  :      
    centroids=csr_matrix((n_clust,matrix.shape[1]))
    for c in cdict:
        cent=0
        n=len(cdict[c])
        for d in cdict[c]:
            cent+=matrix[d]
        cent/=n
        centroids[c]=cent
    return centroids

def getCentroids1(cdict,matrix,n_clust):
    centroids=csr_matrix((n_clust,200000))
    i=0
    for c in cdict:
        cent=0
        n=len(cdict[c])
        for d in cdict[c]:
            cent+=matrix[d]
        cent/=n
        centroids[i]=cent
        i+=1
    return centroids
        

    
def describeCluster(strlist,clusters,centroids,vocab_frame,terms,n_clust):
    content={'contents':strlist,'clusters':clusters}
    frame = pd.DataFrame(content, index = [clusters] , columns = ['contents','clusters'])      
    print('Cluster Counts:')
    print(frame['clusters'].value_counts())
    
    centroids=centroids.toarray()
    centroid=centroids.argsort()[:, ::-1]
    for i in range(n_clust):
            print("Cluster %d words:" % i, end='')
            
            for ind in centroid[i, :10]: 
                print(' %s' % vocab_frame.loc[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
            print() 
            print() 

def eDist(v1,v2):
    return np.sqrt(np.sum(np.square(v1-v2),axis=1))

def predict(M,v):
    return np.argmin(eDist(M,v))

#def recurseCluster(dlist,matrix,n_ind,depth):
#    cdict={}
#    n=len(dlist)
##    print('depth',depth)
##    print('n_dlist',n)
#    nmatrix=matrix[dlist]
#    n_clust=round(n/n_ind)
##    print('n_clust',n_clust)
#    if round(n/n_ind)==1:
#        print('round(n/n_ind)==1',depth)
#        return {0:dlist}
##    print('training kmeans')
#    km=trainKMeans(nmatrix,n_clust)
#    cdict=getClusterDict(km)[1]
#    for c in cdict:
#        for d in range(len(cdict[c])):
#            cdict[c][d]=dlist[cdict[c][d]]
#    l=[s for c in cdict  for s in cdict[c]]
##    print('km clustering done, number of mis matches:',len(set(dlist)-set(l)))
#    if set(dlist)==set(l):
#        test={}
#    #    print('testing',depth)
#        for c in cdict:
#            test[c]=len(cdict[c])<=n_ind
#    #    print(test)
#        if False not in test.values() or depth==0: 
#    #        print('test cleared',depth)
#            return cdict
#        else:
#    #        print('test not cleared',depth)
#            for c in cdict:
#                if not test[c]:
#                    k=cdict[c]
#                    a=recurseCluster(k,matrix,n_ind,depth-1)
#                    a1=reorderDict(dict(squeezeDict(a)))
#                    l=[s for r in a1  for s in a1[r]]
##                    print('number of mis matches:',len(set(k)-set(l)))
#                    if set(k)==set(l):
#                        cdict[c]=a
#    #                print(c,depth,'recurse done')
#                    else:
#                        print('fail')
#            return cdict
#    else:
#        print('problemfound!')
    
def recurseCluster(dlist,matrix,n_ind,depth):
    n=len(dlist)
    n_clust=round(n/n_ind)
#    print('n_clusters',n_clust)
#    print(depth)
    if depth==0 or n_clust<=1:
        return {0:dlist}
    else:
        nmatrix=matrix[dlist]
        if n_clust>10:
#            print('more than 10')
            km=trainKMeans(nmatrix,10)
        else:
#            print('less than equal to 10')
            km=trainKMeans(nmatrix,n_clust)
#        print('km done')
#        km=trainKMeans(nmatrix,n_clust)
        cdict=getClusterDict(km)[1]
#        print('cdict',len(cdict))

        for c in cdict:
            for d in range(len(cdict[c])):
                cdict[c][d]=dlist[cdict[c][d]]  
        l=[s for c in cdict  for s in cdict[c]]
#        adict=cdict.copy()
#        l1=len(adict)
        if set(dlist)==set(l):
            test={}
            for c in cdict:
                test[c]=round(len(cdict[c])/n_ind)<=1
            nxt=[c for c in test if not test[c]]

            for c in nxt:              
                cdict[c]=recurseCluster(cdict[c],matrix,n_ind,depth-1)
                
#            if depth==100:
#               test={}
#               t=[]
#               l2=len(cdict)
#               for c in cdict:
#                   print(c in adict)
#                   if type(cdict[c])!=dict:
#                       t+=cdict[c]
#                       print(c,len(set(adict[c])-set(cdict[c])))
#                   else:
#                       l=dict(squeezeDict(cdict[c]))
#                       l=[s for c in l for s in l[c]]
#                       t+=l
#                       print(c,len(set(adict[c])-set(l)))
#               print(len(set(dlist)-set(t)))
#               print(l1==l2)
#               
            return cdict
        else:
            print('fail')
    

                
def squeezeDict(adict,i=0,n=0):
    tdict=[]
    test=[type(adict[a])!=dict for a in adict]
#    print(test)
    if False not in test:
#        k=0
        for a in adict:
            tdict.append([round(i+a*10**-n,n),adict[a]])
        return tdict
    else:
        for a in adict:
            if type(adict[a])!=dict:
               tdict.append([round(i+a*10**-n,n),adict[a]])
            else:
                tdict+=(squeezeDict(adict[a],round(i+a*10**-n,n),n+1))
        return tdict
    
def reorderDict(adict):
    i=0
    odict={}
    for a in adict:
        odict[i]=adict[a]
        i+=1
    return odict

def getClusters(dlist,matrix,n_ind,depth):
    return reorderDict(dict(squeezeDict(recurseCluster(dlist,matrix,n_ind,depth))))

#def dictElem(adict):
#    l=[s for c in adict for s in adict[c] ]
#    test={}
#    for e in range(len(l)):
#        test[e]=type(l[e])
#    if dict not in list(test.keys()):
#        return l
#    else:
#        t=[e for e in test if test[e]==dict]
#        for e in t:
#            l[e]=dictElem(l[e])
#        return l
    
