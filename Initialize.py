# -*- coding: utf-8 -*-
"""
Created on Fri May 25 12:46:04 2018

@author: C63165
"""

from Indexer import processJSON,ListJSON
from Clusterer import importData,vectorize,buildVocabFrame
from Doc2vec_Clusterer import labelSent,d2v
from Doc2vec_search import getClusters
import pickle

#
# list_json=['AaronPressman']
list_json=['bbc/business','bbc/entertainment','bbc/politics','bbc/sport','bbc/tech']
# list_json=['business','tech']
obj=ListJSON(list_json)
files=['json/filesearch.json']
ddict=processJSON('json/processed.json',files)


file=open('clusters/ddict.sav','wb')
pickle.dump(ddict,file)
file.close()

data=importData('json/processed.json')
doclist=data[0]
strlist=data[1]

file=open('clusters/doclist.sav','wb')
pickle.dump(doclist,file)
file.close()

file=open('clusters/strlist.sav','wb')
pickle.dump(strlist,file)
file.close()

tagged_content=labelSent(strlist)

file=open('clusters/tagged.sav','wb')
pickle.dump(tagged_content,file)
file.close()

model=d2v(tagged_content)

file=open('clusters/doc2vec_model.sav','wb')
pickle.dump(model,file)
file.close()

tfidfmodel=vectorize(strlist)

file=open('clusters/tfidfmodel.sav','wb')
pickle.dump(tfidfmodel,file)
file.close()

vocab_frame=buildVocabFrame(strlist)

file=open('clusters/vocab_frame.sav','wb')
pickle.dump(vocab_frame,file)
file.close()

n=3 #number of documents required ideally in a cluster
cdict=getClusters(list(range(len(doclist))),tfidfmodel[1],n,100)

file=open('clusters/cdict.sav','wb')
pickle.dump(cdict,file)
file.close()