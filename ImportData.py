 # -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:07:45 2018

@author: C63165
"""
from Indexer import Document
from Indexer import Document1
from Indexer import Documents
from Indexer import processJSON
import time

def importData(dest,docs,files,overwrite=True):
 """
 Imports the data from the json files retrieved from mongoDB.
 inputs:
     dest- destination file name for storing processed json (string).
     docs- Documents Object to which the documents should be added.
     files- list of json file names which were exported from mongoDB.
     overwrite- whether or not the dest file should be overwritten (boolean).
 outputs:
     a tuple with first element as the dictionary from processJSON() and second element as the updated Documents Object.
 """
 t0=time.time()
 print('Initiallizing import...')
 ddict=processJSON(dest,files,overwrite)
 i=1
 for d in ddict:
   try: 
    doc=Document1(ddict[d]["_id"]["$oid"],{})
    if doc not in docs.docs:
     doc=Document(ddict[d])
     print('Importing Document No: '+str(i)+'...')
     docs.addDoc(doc)
     i+=1
   except:
       continue
 print('Importing Done!')

 docs.saveDocs()
 docs.exportToJSON()
 docs.exportDocID()
 t1=time.time()
 print('Time Taken: '+str(t1-t0))
 return (ddict,docs)

docs=Documents('Docs')
imported=importData('processed.json',docs,['testfilesearch.json'])
#imported=importData('processed.json',docs,['crawleq.json'])
#imported=importData('processed.json',docs,['crawl.json'],False)
#imported=importData('processed.json',docs,['crawl1.json'],False)
#imported=importData('processed.json',docs,['crawl4.json'],False)
#imported=importData('processed.json',docs,['crawl3.json'],False)
ddict=imported[0]
docs=imported[1]