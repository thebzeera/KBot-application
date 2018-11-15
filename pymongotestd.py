import ntpath
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from flask_cors import CORS
from flask import Flask, jsonify, request, send_from_directory
from flask_pymongo import PyMongo
from fuzzywuzzy import fuzz
from spell_checker import correction
# from flask_socketio import SocketIO
import bson
import json
import string
import nltk, re, pprint
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer

ps = PorterStemmer()
# import spacy
from nltk.stem import WordNetLemmatizer
from Clusterer import Cluster, tokenize_and_stem
from sklearn.metrics.pairwise import cosine_similarity
from Doc2vec_search import getCentroids, loadModel
import pickle
import os
# import spellcheck_c
import gensim
import re
from gensim.summarization.summarizer import summarize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy

lm = WordNetLemmatizer()
output1 = []

username = None
iter_n = 0
doc = []
bestmatch = []
output2 = ""
# global doc_id

tfidf, strlist, vocab_frame = loadModel('clusters/tfidfmodel.sav', 'clusters/strlist.sav', 'clusters/vocab_frame.sav')
file = open('clusters/doclist.sav', 'rb')
doclist = pickle.load(file)
file.close()
file = open('clusters/ddict.sav', 'rb')
ddict = pickle.load(file)
file.close()
file = open('clusters/doc2vec_model.sav', 'rb')
model = pickle.load(file)
file.close()
file = open('clusters/cdict.sav', 'rb')
cdict = pickle.load(file)
file.close()
clusters = list(range(len(doclist)))
for c in cdict:
    for d in cdict[c]:
        clusters[d] = c


def paraclust(texts, query):
    print("__________texts_________", texts)
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)
    # print(X,"fittt")

    words = vectorizer.get_feature_names()
    print("words", words)
    print(len(texts))

    n_clusters = 5
    number_of_seeds_to_try = 50
    max_iter = 300
    number_of_process = 1
    from sklearn.cluster import MeanShift
    # model = MeanShift(bandwidth=2,n_jobs=number_of_process).fit(X)
    model = KMeans(n_clusters=n_clusters, max_iter=max_iter, n_init=number_of_seeds_to_try, n_jobs=number_of_process,
                   random_state=1234).fit(X)

    labels = model.labels_

    ordered_words = model.cluster_centers_.argsort()[:, ::-1]

    print("centers:", model.cluster_centers_)
    print("labels", labels)
    print("intertia:", model.inertia_)

    texts_per_cluster = numpy.zeros(n_clusters)
    for i_cluster in range(n_clusters):
        for label in labels:
            if label == i_cluster:
                texts_per_cluster[i_cluster] += 1

    print("Top words per cluster:")
    for i_cluster in range(n_clusters):
        print("Cluster:", i_cluster, "texts:", int(texts_per_cluster[i_cluster])),
        for term in ordered_words[i_cluster, :10]:
            print("\t" + words[term])

    print("\n")
    print("Prediction")

    text_to_predict = query.lower()
    print("texttt", text_to_predict)
    vectorizer.transform([text_to_predict])
    # vectorizer.vocabulary_
    Y = vectorizer.transform([text_to_predict])

    predicted_cluster = model.predict(Y)[0]
    print(predicted_cluster, "predictedn  cluster")
    texts_per_cluster[predicted_cluster]

    print(text_to_predict)
    print("Cluster:", predicted_cluster, "texts:", int(texts_per_cluster[predicted_cluster])),
    term_list=[]
    for term in ordered_words[predicted_cluster, :10]:
        print("\t" + words[term])
        term_list.append(words[term])

    paralist = []
    if (query in term_list):
        k = 0
        for i in labels:
            if (i == predicted_cluster):
                paralist.append(k)
            k += 1
    else:
        [paralist.append(i) for i, x in enumerate(texts)]
        print(paralist)
    # exit()

    # exit()
    # print(term_list,"term_list")
    # for terms in term_list:
    #     n = 0

            # print("enter")
            #
            # for text in texts:
            #     para=text
            #     paralist.append(n)
        # n+=1
    # print(paralist)
    return paralist


# clusters,cdict=getClusterDict(km)
#
# centroids=getCentroids(cdict,tfidf[1],len(cdict))
#
# describeCluster(strlist,clusters,centroids,vocab_frame,tfidf[2],len(cdict))

cluster = Cluster(doclist, cdict, ddict, tfidf[0], tfidf[1])
# print(v)

def symmetric_sentence_similarity(sentence1, sentence2):
    return (fussymatch(sentence1, sentence2))

def fussymatch(sentence1, sentence2):
    return fuzz.token_set_ratio(sentence1, sentence2)

        # return (sentence_similarity(sentence1, sentence2) + sentence_similarity(sentence2, sentence1))


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


def cosine_sentence_similarity(sentence1, sentence2):
    """

    :param sentence1:
    :param sentence2:
    :return:
    """
    # print("cosine similarity")
    sent1 = sentence1
    sent2 = sentence2

    li = []
    li.append(sent1)
    li.append(sent2)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(li)
    v = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
    # print("vvvvvvvvvvv",v)
    for c in v:
        value = sorted(c)
        # print("ooooo",value)
        value = value[-2]
    return value


def sentence_similarity(sentence1, sentence2):
    sent1 = sentence1
    sent2 = sentence2

    sentence1 = pos_tag(word_tokenize(sent1))

    sentence2 = pos_tag(word_tokenize(sent2))

    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    # print(synsets1,"tttttt")
    # print(sentence1,"vvvv")
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]

    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]

    score, count = 0.0, 0

    for synset in synsets1:
        # print("synset simmilarity")

        high = [synset.path_similarity(ss) for ss in synsets2 if synset.path_similarity(ss) != None]
        best_score = 0

        if len(high) != 0:
            best_score = max(high)

        if best_score is not None:
            score += best_score
            # print("score",score)
            count += 1

    if count > 0:

        score /= count

    if score == 0:
        score = cosine_sentence_similarity(sent1, sent2)
        # print("cosinescore",score)


    return score


app = Flask(__name__, static_url_path='')
CORS(app)

app.config['MONGO_DBNAME'] = 'vor'
app.config['MONGO_URI'] = 'mongodb://127.0.0.1:27017/vor'

mongo = PyMongo(app)


def get_nested(data, *args):
    if args and data:
        element = args[0]
        if element:
            value = data.get(element)

        return value if len(args) == 1  else get_nested(value, *args[1:])


def addUser(username, user_db, query):
    print('at add user')
    userToAdd = {
        "name": username,
        "query_list": [],
        "cluster": []

    }
    user_db.insert_one(userToAdd)
    print('user added:', userToAdd)

    return True


rel_para = [0, 0, 0]


# @socketio.on('relevance')
# def printer(rel_para):

# print('relevance received')
# print(rel_para)

@app.route("/")
def index():
    return app.send_static_file('index.html')

    # return app.send_static_file('index.html')


@app.route('/relevance', methods=['POST'])
def printer():
    print("somethingg")
    print('id is', doc_id)
    print(output1)
    data = request.get_json(force=True)
    print(data)
    import ast
    data = ast.literal_eval(data)
    print(type(data))
    # print('here is data',rel_para,username,msg)
    print('here is data', data)
    print(type(data))
    user = mongo.db.userdata
    val = user.find_one({"name": data['username'], "query": data['name'].lower()})

    if val == None:
        emp_rec1 = {
            "name": data['username'],
            "docchoice": data['docId'],
            "query": data['name'].lower(),
            "parachoice": data['rel_para'],
            "output": output2
        }
        user.insert_one(emp_rec1)

    user.update_many(
        {"name": data['username'], "query": data['name'].lower()},
        {
            "$set": {
                "docchoice": data['docId'],
                "parachoice": data['rel_para'],
                "output": output2
            },
            "$currentDate": {"lastModified": True}

        }
    )
    return jsonify({"success": True})


# @socketio.on('username')


# def usersearch(username):
#     print('username received')
#     user_db=mongo.db.user_db
#     user_data=user_db.find_one({"name": username})
#     if user_data==None:
#         addUser(username,user_db)
#     else:
#         print(user_data)
#     return user_data

global flag


# @app.route('/input/<query>', methods=['POST'])
@app.route('/input/<query>', methods=['POST'])
def get_all_frameworks(query):
    try:
        data = request.get_json(force=True)
        print(data)

        import ast
        data = ast.literal_eval(data)
        print(data,"data")
        user = mongo.db.userdata
        val = user.find_one({"name": data['username'], "query": data['query'].lower()})
        print("val",val)
        print(cluster.clustrec(data['query'].lower(), 'e'))
        global doc_id
        if val == None:
            userdb = mongo.db.user_db
            vall = userdb.find_one({"name": data['username']})
            # userdb.update({"name":data['username']},{"$addToSet":{"query_list":query.lower}})
            if vall == None:
                addUser(data['username'], userdb, query)

                # userdb.update_many(
            # {"name":data['username'], "query_list":data['query'].lower()})
            userdb.update(
                {"name": data['username']},
                {
                    "$addToSet": {"query_list": data['query'].lower()}
                })
            # print(query,"/||||||| query|||||")
            # if data['query'].isalnum():
            #     data['query']="what is "+data['query']
            # print(data['query'], "rrrr")
            lower_query = query.lower()
            iter_n = int(lower_query[-1])
            name = lower_query[:-1]
            ''' 
            checker = spellcheck_c.min_edit_distance(name)
            print('spellcheck out', checker)
            if checker != 1:
                out="Sorry, did you mean "
                out+= checker
                out+= ' ?'
                output1=[]
                output1.append(out)  
                output1.append(3)      
                # socketio.emit('correction',output1)
                name = checker
            '''
            IDs = cluster.clustSearch(name, 'e')
            print("-----IDS-------")
            print(IDs)
            print("-----iter_n-------")
            print(iter_n)
            paras = cluster.getParaList(IDs[iter_n])
            print("-----Paras-------")
            print(paras)
            print("clustering")
            paralist = paraclust(paras, name)
            print("paralist", paralist)
            # exit()
            # global doc_id
            doc_id = IDs[iter_n]
            print(doc_id)

            print("-----doc ID-------")
            # print(doc_id)
            # print(paras)
            b = 0
            score = 0
            best1 = 0
            para = 0
            best3 = 0
            para3 = 0
            para2 = 0
            best2 = 0
            content_length = len(paras)
            # print(symmetric_sentence_similarity(name, paras[5]))
            for b in paralist:
                mylist = paras[b]
                # print("mylist",mylist)
                if len(mylist) > 10:
                    # print(mylist)
                    score = symmetric_sentence_similarity(name, mylist)
                    # print("SCOREprint",b,score,mylist)
                    if score > best1:
                        best3 = best2
                        best2 = best1
                        best1 = score
                        para3 = para2
                        para2 = para
                        para = b
                    elif score > best2:
                        best3 = best2
                        best2 = score
                        para3 = para2
                        para2 = b
                    elif score > best3:
                        best3 = score
                        para3 = b

            print('para is:', para)
            print('para2 is:', para2)
            print('para3 is:', para3)
            score = symmetric_sentence_similarity(name, paras[0])
            if para3==0 and para2==0 and para==0 and score==0:
                para2=1
                para3=2
            elif para2==0 and para3==0 and score==0:
                para2=para+1
                para3=para2+1
            elif para3==0 and score==0:
                para3=para2+1
            output1 = []
            output1.append(paras[para])
            output1.append(paras[para2])
            output1.append(paras[para3])
            # if para!=0:
            #
            #     output1.append(paras[para2])
            #     if para3 != 0:
            #         output1.append(paras[para3])
            # if para != 0:
            #     output1.append(paras[para2])
            #     output1.append(paras[para3])

            print('output1 is:', output1[0])
            ##output=[]
            # output2= output1[iter_n]
            # my_lst_str = ''.join(map(str, output1))
            # print(summarize(my_lst_str))

            # socketio.emit('output',output1)
            # socketio.emit('imageConversionByServer',img64)
            return jsonify(output1, doc_id)
        else:
            userdb = mongo.db.user_db
            userdb.update(
                {"name": data['username']},
                {
                    "$addToSet": {"query_list": data['query'].lower()}
                })
            name = query[:-1]
            output2 = []
            output1 = []
            print(val['docchoice'])
            paras = cluster.getParaList(val['docchoice'])
            flag = 1
            paralist = paraclust(paras, name)
            b = 0
            # global doc_id
            score = 0
            best1 = 0
            para = 0
            best3 = 0
            para3 = 0
            para2 = 0
            best2 = 0
            print(len(paras))
            content_length = len(paras)
            print(symmetric_sentence_similarity(name, paras[0]))
            for b in paralist:
                mylist = paras[b]

                if len(mylist) > 10:
                    score = symmetric_sentence_similarity(name, mylist)
                    if score > best1:
                        print('in if')
                        print('score is:', score)
                        best3 = best2
                        best2 = best1
                        best1 = score
                        para3 = para2
                        para2 = para
                        para = b
                        print('para2 at b=', b, ' is :', para2)
                        print('para3 at b=', b, ' is :', para3)

            print('para is:', para)
            print('para2 is:', para2)
            print('para3 is:', para3)

            output2.append(paras[para])
            if para != 0:
                output2.append(paras[para2])
                output2.append(paras[para3])

            # paras=cluster.getParaList()
            choice = val['parachoice'].split(",")
            pg = 0
            i = 0
            print(output2)
            while i < len(output2):
                # print(output2[pg])
                print(choice[i])
                if (choice[i] == "1"):
                    output1.append(output2[pg])
                    print(output2[pg])
                pg += 1
                i += 1

                # print(output1[pg])

            # print summarize(output1[0])
            print(output1)

            return jsonify(output1, doc_id)
    except:
        output1 = []
        output1.append("Oops ! Sorry, I am unable to answer to this query. ")
        return jsonify(output1)


def getSyn(word):
    syns = []
    for synset in wn.synsets(word):
        syns = syns + synset.lemma_names()
    syns = set([w.lower() for w in syns])
    return syns



@app.route('/downloads', methods=['GET'])
def download_file():
    print("something")
    print(request.query_string, "dociddd")
    json = request.args.getlist('doc_id')
    for i in json:
        # print(filename,"file nameee")
        filename = ntpath.basename(i)
        head, tail = os.path.split(i)
        # print(filename)
        # print(app.static_folder)
        path = os.path.dirname(os.path.realpath(filename))
        print(path, "dir path")
        full_path = path + '\\' + head
        # print(full_path,"full_path")
        # # p=path+'\yahoo'
        # v=os.path.realpath(filename)
        # print(v,"vvv")
        # head, tail = os.path.split(v)
        # print(tail,"tail")
        # print(head,"head")
        # path = os.path.dirname(os.path.realpath(filename))
        # p = path + '\data'
        # print(p)
        # directory = os.path
        # print(directory)
        # print(os.path.join(app.instance_path, ''))
        # temp=send_from_directory(p, filename, as_attachment=True)

        #
        return send_from_directory(full_path, filename, as_attachment=True)

@app.route('/inputs/<query>', methods=['GET'])
def spellchecker(query):
    listBoth={}

    print(request.query_string)
    json = request.args.getlist('dataa')
    print(json)
    for i in json:
        query=i
    # # print(query,"fgdf")
    # # print(correction(query),"vvvv")

    corrected=correction(query)
    print(query," is corrected to ",corrected)
    if (corrected==query):
        return jsonify(query,query)
    else:
        return jsonify(query,corrected)






if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)