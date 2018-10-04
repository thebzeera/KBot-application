import json
import math
import os
import spacy
#import numpy as np
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
#from gensim.summarization import summarize
#from gensim.summarization import keywords
ps = PorterStemmer()
lm=WordNetLemmatizer()

def ListJSON(file):
    """

        :param file: path of the dataset directory
        :return: JSON file
        """
    file_path=file

    with open('json/filesearch.json', 'w', )as json_file:
        print('')

    for i in file_path:
        add_slash = os.path.join(i, '', '')
        for filename in os.listdir(i):
            content = {}
            id = {}
            full_content = {}
            temp = []
            file_path = add_slash + filename
            with open(file_path, 'r') as text:
                text=text.read()
                # nlp = spacy.load('en_coref_md')
                # doc = nlp(text)
                # doc_coref=doc._.coref_resolved
                # te=text.splitlines()
                splat = text.split("\n\n")
                # print(splat)
                p = 0
                header = ''
                for number, paragraph in enumerate(splat, 1):

                    if len(paragraph) < 70:
                        header = paragraph
                    else:
                        combine = header + paragraph
                        temp.insert(number, combine)

                content["contents"] = temp
                full_content["_id"] = id
                id["$oid"] = file_path
                full_content["content"] = content
                    # # document["D" + str(i)] = full_content
            # i = i + 1
            with open('json/filesearch.json', 'a'  ) as json_file:
                json.dump(full_content, json_file)
                json_file.write('\n')

    return json_file
          
def processJSON(dest,files,overwrite=True):
    """
    Processed the JSON file which was exported from mongoDB and returns a valid JSON file.
    Inputs:
        dest- Name of the destination file (string)
        files- list of JSON file names which were exported from mongoDB (list)
        overwrite- whether or not the destination file should be overwritten. (boolean)
    Outputs:
        writes a JSON file with the name dest
        returns a python dictionary corresponding to the JSON file
    """
    rawlist=[]
    i=1
    for file in files:
#     print('new-file')
     crawl=open(file,'r',encoding='utf8')
     r=crawl.readline()
#     print("while-in")
     while r:
         rawlist.append(r)
         r=crawl.readline()
         print('Processed Document: '+str(i))
         i+=1
        
    start=0  
    js1='{}'
    if not overwrite:
      file1=open(dest,'r',encoding='utf8')
      js1=file1.read()
      file1.close()
      js2=json.loads(js1)
      start=max([int(i[1:]) for i in js2])+1
      js1=js1[:-1]+','+'}'
#    print('hail')
    js='{'
    for i in range(len(rawlist)):
     js=js+'"d'+str(i+start)+'":'+rawlist[i]+','
    js=js[:-1]+'}' 
#    print('hydra')
    print('Writing Processed json file')
    jsf=js1[:-1]+js[1:]
    file1=open(dest,'w',encoding='utf8')
    file1.write(jsf)
    file1.close()
    
    return json.loads(jsf)

class Document(object):
    """
    A class which represents an individual document.
    """
    def __init__(self,js):
        """
        input:
            js- a dictionary which corresponds to a document. This is obtained from the processed json.
        output:
            initializes a Document object.
        """
        self.ID=js["_id"]["$oid"]
        self.stringlist=js["content"]["contents"]
        self.string='\n'.join(self.stringlist)
        r=''
        string=self.string
        # for i in '!@#$%^&*()_+-={}[]|\\\';:"<,>.?/':
        #    r=string.split(i)
        #    self.string=' '.join(r)
        string=string.lower()
        self.wordlist=string.split()
        self.wordlist=[ps.stem(lm.lemmatize(w)) for w in self.wordlist if len(w)>2 ]
        self.TF={}
        self.computeTF()
        
    
    def getWordCount(self):
        """
        returns the word count of the document (excluding the stop words).
        """
        return len(self.wordlist)
    
    def getWordList(self):
        """
        returns a list containing the words in the document (excluding the stop words).
        """
        return self.wordlist
    
    def getCountOf(self,word):
        """
        returns the number of occurances of the word in the document.
        """
        return self.wordlist.count(word)
    
    def isWordIn(self,word):
        """
        returns True if the word is in document, False otherwise.
        """
        return word in self.wordlist
    
    def getTF(self,word):
        """
        returns the term frequency (TF) of the word in the document.
        """
        return float(self.getCountOf(word))/float(self.getWordCount())
    
    def computeTF(self):
        for word in self.wordlist:
            self.TF[word]=self.getTF(word)
    
    def getID(self):
        """
        returns the ID of the document (which is the object id of the document in mongoDB).
        """
        return self.ID
    
    def __eq__(self,doc):
        """
        condition for checking equivalence of two documents.
        Two documents are equal if their ID's are the same.
        """
        return self.ID==doc.ID

class Document1(Document):
    def __init__(self,ID,TF):
        self.ID=ID
        self.wordlist=list(TF.keys())
        self.TF=TF
    
class Documents(object):
    """
    A class which represents a collection of Documents.
    """
    def __init__(self,name):
        """
        Initializes a Documents object. Checks if Documents object with same name is already saved. 
        Restores from previous checkpoint. Else initialises a new Documents object
        """
        try:
            file=open(name+'.json','r')
            js=file.read()
            file.close()
            docs=json.loads(js)
            self.name=docs['name']
            self.docs=[]
            for d in docs['docs']:
                doc=Document1(d,docs['docs'][d])
                self.docs.append(doc)
            self.wordlist=set(docs['IDF'].keys())
            self.idf=docs['idf']
            self.idf=docs['IDF']
            file=open('TFIDF.json','r')
            js=file.read()
            file.close()
            self.TFIDF=json.loads(js)
            self.summary=docs['summary']
            self.keywords=docs['keywords']
            self.updated=True
        except:
         self.name=name
         self.docs=[]
         self.wordlist=set({})
         self.idf={}
         self.TFIDF={}
         self.updated=True
         self.summary={}
         self.keywords={}
         
        stopfile=open('stopwords.txt','r',encoding='utf8')
        self.stopwords=stopfile.read()
        self.stopwords=self.stopwords.split()
        self.IDF={}
#        self.numDocs=self.getNumDocs()
#        self.queue=set({})
        
    
    def getNumDocs(self):
        """
        returns the number of documents in the collection.
        """
        return len(self.docs)
    
    def updateIDF(self):
        """
        Computes the IDF of all of the words in the wordlist of the collection and stores the values in self.IDF (a dictionary).
        """
        for word in self.wordlist:
            self.IDF[word]=math.log(self.getNumDocs()/self.idf[word])
    
    def computeTFIDF(self):
        """
        Computes the TFIDF of all the words for each of the document in the collection and stores the values in self.TFIDF (a dictionary of dictionaries).
        """
        for doc in self.docs:
            for word in set(doc.wordlist):
                try:
                    self.TFIDF[word][doc.getID()]=doc.TF[word]*self.IDF[word]
                except:
                    try:
                        tf=doc.TF[word]
                        self.TFIDF[word]={}
                        self.TFIDF[word][doc.getID()]=tf*self.IDF[word]
                    except:
                        continue
                    
    def getTFIDF(self,doc,word):
        try:
            return self.TFIDF[ps.stem(lm.lemmatize(word))][doc.getID()]
        except:
            return 0
        
    def addDoc(self,doc):
        """
        Adds a document to the collection and updates the wordlist.
        input:
            doc- a Document object.
        """
        if doc not in self.docs:
            self.docs.append(doc)
            for word in set(doc.getWordList()):
                    if word not in self.stopwords:
                      self.wordlist.add(word)
#                      self.queue.add(word)
                      if word in self.idf:
                          self.idf[word]=self.idf[word]+1
                      else:
                          self.idf[word]=1
                    
            self.updated=False

    def getMatch(self, words,n=1):
        """
        Returns the IDs of the n documents which corresponds to the best match for the list of words
        input:
            words- a list of words.
        """
        match={}
        for doc in self.docs:
            s=0
            flag=True
            for word in words:
#                try:
#                    t=self.getTFIDF(doc,word)
#                    if t==0:
                t=[self.getTFIDF(doc,w) for w in getSyn(word)]
#                        print(t)
#                except:
#                    t=self.getTFIDF(doc,word)
                if not t:
                    t=[self.getTFIDF(doc,word)]
                t=sum(t)
                if t==0:
                  flag=False
                  break
                s=s+t
            if flag:
                match[s]=doc.getID()
        best=sorted(match)
        best.reverse()
        best=[b for b in best[0:n] if b>0]
        bestmatch=[match[b] for b in best]
        return bestmatch
      
    def summarize(self):
      for doc in self.docs:
         try:
          self.summary[doc.getID()]=summarize(doc.string,word_count=60)
         except:
             continue
    
    def getKeywords(self):
        for doc in self.docs:
         try:
            self.keywords[doc.getID()]=keywords(doc.string,words=10,split=True,lemmatize=True)
         except:
             continue
    
    def update(self):
      """
      Updates self.IDF and self.TFIDF
      """
      if not self.updated:
        print('Computing IDF...')
        self.updateIDF()
        print('Computing IDF done!')
        print('Computing TFIDF...')
        self.computeTFIDF()
        print('Computing TFIDF Done!')
#        self.summarize()
#        self.getKeywords()
        print('Finished Updating!')
        self.updated=True
#        self.queue=set()
    
    def exportToCSV(self):
        """
        Exports the TFIDF data to a csv file.
        """
        print('Exporting to TFIDF.csv...')
        file=open('TFIDF.csv','w',encoding='utf8')
        file.write('Word,')
        for doc in self.docs:
            file.write(doc.getID()+',')
        file.write('\n')
        for word in self.wordlist:
            file.write(word+',')
            for doc in self.docs:
              if word in doc.getWordList():
                file.write(str(self.TFIDF[word][doc.getID()])+',')
              else:
                 file.write(str(0)+',') 
            file.write('\n')
        file.close()
        print('Exporting Done!')
    
    def exportToJSON(self):
        """
        Exports the TFIDF data to a json file.
        """
        print('Exporting to TFIDF.json...')
        jsonstr=json.dumps(self.TFIDF)
        file=open('TFIDF.json','w',encoding='utf8')
        file.write(jsonstr)
        file.close
        print('Exporting Done!')
    
    def saveDocs(self):
        if not self.updated:
            self.update()
        save={}
        save['name']=self.name
        save['docs']={}
        for doc in self.docs:
            save['docs'][doc.getID()]=doc.TF
        save['idf']=self.idf
        save['IDF']=self.IDF
        save['summary']=self.summary
        print(save['summary'],"summary")
        save['keywords']=self.keywords
       # save['ddict']=self.ddict
        savejs=json.dumps(save)
        file=open(self.name+'.json','w',encoding='utf8')
        file.write(savejs)
        file.close()
        return save
    
    def saveSumKey(self):
        save={}
        save['summary']=self.summary
        save['keywords']=self.keywords
        savejs=json.dumps(save)
        file=open(self.name+'_sumkey.json','w',encoding='utf8')
        file.write(savejs)
        file.close()
        return save
    
    def indvsumkey(self):
        i=1
        for ID in self.summary:
            js={}
            file=open(str(i)+'.json','w',encoding='utf8')
            js['ID']=ID
            js['summary']=self.summary[ID]
            js['keywords']=self.keywords[ID]
            js=json.dumps(js)
            file.write(js)
            file.close()
            i+=1
    
    def getDocID(self):
        """
        Returns a list of IDs of all the documents in the collection.
        """
        ID=[]
        for doc in self.docs:
            ID.append(doc.getID())
        return ID
    
    def exportDocID(self):
        """
        Exports the list of Document IDs and stores it in a text file.
        """
        ID=open('ID.txt','w',encoding='utf8')
        ID.write(str(self.getDocID()))
        ID.close()
def getPara(ddict,docs,words,n1,n2):
    """
    Returns the n2 th paragraph from the n1 documents corresponding to the best match for the given words.
    input:
        ddict:The dictionary which was returned from processJSON().
        docs: a Documents object.
        words: a list of words.
        n1: number of matches(int)
        n2: paragraph number(int).
    output:
        nth paragraph (string).
    """
    IDs=docs.getMatch(words,n1)
    para=[]
    for ID in IDs:
     for d in ddict:
        if ddict[d]["_id"]["$oid"]==ID:
            para.append(ddict[d]['content']['contents'][n2])
    return para

def getIntro(ddict,docs,words,n1):
    """
    Returns the 1st paragraph from the n1 documents corresponding to the best match for the given words.
    input:
        ddict:The dictionary which was returned from processJSON().
        docs: a Documents object.
        words: a list of words.
        n1: number of matches(int)
        
    output:
        first paragraph (string).
    """
    return getPara(ddict,docs,words,n1,0)
        
def getSyn(word):
    syns= []
    for synset in wordnet.synsets(word):
        syns=syns+synset.lemma_names()
    syns=set([w.lower() for w in syns])
    return syns




