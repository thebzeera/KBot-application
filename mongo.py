from flask import Flask, jsonify, request
from flask_pymongo import PyMongo
from flask_socketio import SocketIO
import bson
import json
import string
import nltk, re, pprint
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
ps = PorterStemmer()

from nltk.stem import WordNetLemmatizer
from Clusterer import Cluster,tokenize_and_stem
from Doc2vec_search import getCentroids,loadModel
import pickle
import spellcheck_c
lm=WordNetLemmatizer()


username=None
iter_n=0
doc=[]
bestmatch=[]

tfidf,strlist,vocab_frame=loadModel('tfidfmodel.sav','strlist.sav','vocab_frame.sav')
file=open('doclist.sav','rb')
doclist=pickle.load(file)
file.close()
file=open('ddict.sav','rb')
ddict=pickle.load(file)
file.close()
file=open('doc2vec_model.sav','rb')
model=pickle.load(file)
file.close()
file=open('cdict.sav','rb')
cdict=pickle.load(file)
file.close()
clusters=list(range(3226))
for c in cdict:
    for d in cdict[c]:
        clusters[d]=c
        
#clusters,cdict=getClusterDict(km)
#            
#centroids=getCentroids(cdict,tfidf[1],len(cdict))
#
#describeCluster(strlist,clusters,centroids,vocab_frame,tfidf[2],len(cdict))
cluster=Cluster(doclist,cdict,ddict,tfidf[0],tfidf[1])
def symmetric_sentence_similarity(sentence1, sentence2):
    
    return (sentence_similarity(sentence1, sentence2) + sentence_similarity(sentence2, sentence1)) 
def penn_to_wn(tag):
    
    if tag.startswith('N'):
        return 'n'
 
    if tag.startswith('V'):
        return 'v'
 
    if tag.startswith('J'):
        return 'a'
 
    if tag.startswith('R'):
        return 'r'
 
    return None
 
def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None
 
    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None
 
def sentence_similarity(sentence1, sentence2):
    
   
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))
 
   
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
 
    
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]
 
    score, count = 0.0, 0
    
    
    for synset in synsets1:
        
        high= [synset.path_similarity(ss) for ss in synsets2 if synset.path_similarity(ss)!=None]
        best_score=0
       

        if len(high) !=0 :
            
                
                best_score = max(high)
 
        
        if best_score is not None:
            score += best_score
            count += 2
 
    
    if count > 0 :
        score /= count
    return score
 


app = Flask(__name__)

socketio = SocketIO(app)

app.config['MONGO_DBNAME'] = 'testfilesearch'
app.config['MONGO_URI'] = 'mongodb://127.0.0.1:27017/vor'



mongo = PyMongo(app)

def get_nested(data, *args):
      if args and data:
        element  = args[0]
        if element:
            
            value = data.get(element)
            
             
        return value if len(args) == 1  else get_nested(value, *args[1:]) 


def addUser(username,user_db):
    print('at add user')
    userToAdd={
        "name":username,
        "query_list": [],
        "category": []

    }
    user_db.insert_one(userToAdd)
    print('user added:',userToAdd)
    
    return usersearch(username)
rel_para=[0,0,0]

#@socketio.on('relevance')
#def printer(rel_para):
   
   # print('relevance received')
    #print(rel_para)

        
@socketio.on('username')

def usersearch(username):
    print('username received')
    user_db=mongo.db.user_db
    user_data=user_db.find_one({"name": username})    
    if user_data==None:
        addUser(username,user_db)
    else:
        print(user_data)
    return user_data
        
      

@socketio.on('input')
def get_all_frameworks(name,iter_n,username):
    print(type(name))
    
    checker = spellcheck_c.min_edit_distance(name)
    print('spellcheck out', checker)
    if checker != 1:
        out="Sorry, did you mean "
        out+= checker
        out+= ' ?'
        output1=[]
        output1.append(out)  
        output1.append(3)      
        socketio.emit('correction',output1)
        name = checker
        
    IDs=cluster.clustSearch(name,'e')
    print(IDs)
    paras=cluster.getParaList(IDs[iter_n])
    print(iter_n)
    #print(paras)
    b=0
    score=0
    best1=0
    para=0
    best3=0
    para3=0
    para2=0
    best2=0
    content_length=len(paras)
    #print(symmetric_sentence_similarity(name, paras[5]))
    while b <  content_length:
            mylist = paras[b]
            
            if len(mylist) > 10:
                score=symmetric_sentence_similarity(name, mylist)
                if score > best1 :
                    print('in if')
                    print('score is:',score)
                    best3=best2
                    best2=best1
                    best1=score
                    para3=para2
                    para2=para
                    para=b
                    print('para2 at b=',b,' is :',para2)
                    print('para3 at b=',b,' is :',para3)
            b+=1

    
    
    print('para is:',para)
    print('para2 is:', para2)
    print('para3 is:',para3)
    
    output1=[]
    output1.append(paras[para])
    if para!=0:

        output1.append(paras[para2])
        output1.append(paras[para3])
        
    print('output1 is:',output1[0]) 
        
    user=mongo.db.userdata
    val=user.find_one({"name":username,"query":name.lower()})
    @socketio.on('relevance')
    def printer(rel_para):
        print (rel_para)
        user=mongo.db.userdata
        val=user.find_one({"name":username,"query":name.lower()})
    
        if val==None :
            emp_rec1 = {
            "name":username,
            "docchoice":doc[bestmatch[iter_n]],
            "query":name.lower(),
            "parachoice":rel_para
            }
            user.insert_one(emp_rec1)


        user.update_many(
            {"name":username, "query":name.lower()},
            {
                "$set":{
                        "docchoice":doc[bestmatch[iter_n]],
                        "parachoice":rel_para
                        },
                "$currentDate":{"lastModified":True}
                
                }
        )         
    
    #if val==None :
        #emp_rec1 = {
        #"name":username,
        #"docchoice":doc[bestmatch[iter_n]],
    # "query":name.lower(),
    # "parachoice":rel_para
        #}
        #user.insert_one(emp_rec1)


    #user.update_many(
        #{"name":username, "query":name.lower()},
        #{
                #"$set":{
                        #"docchoice":doc[bestmatch[iter_n]],
                        #"parachoice":rel_para
                    # },
            # "$currentDate":{"lastModified":True}
                
            # }
        #)
    #socketio.emit('imageConversionByServer',img64)    
    output=[]
    
    socketio.emit('output',output1)
    #socketio.emit('imageConversionByServer',img64)
    return jsonify({'search result' : output})


      

def getSyn(word):
    syns= []
    for synset in wn.synsets(word):
        syns=syns+synset.lemma_names()
    syns=set([w.lower() for w in syns])
    return syns




if __name__ == '__main__':
    socketio.run(app)
