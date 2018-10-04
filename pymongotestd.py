from flask_cors import CORS
from flask import Flask, jsonify, request
from flask_pymongo import PyMongo
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
from Doc2vec_search import getCentroids, loadModel
import pickle
# import spellcheck_c
import gensim
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
global doc_id

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
    print("++++++++++++++++++++++++++++++")
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)

    words = vectorizer.get_feature_names()
    print("words", words)
    print(len(texts))

    n_clusters = 5
    number_of_seeds_to_try = 10
    max_iter = 300
    number_of_process = 1
    model = KMeans(n_clusters=n_clusters, max_iter=max_iter, n_init=number_of_seeds_to_try,
                   n_jobs=number_of_process).fit(X)

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

    text_to_predict = query
    Y = vectorizer.transform([text_to_predict])
    predicted_cluster = model.predict(Y)[0]
    texts_per_cluster[predicted_cluster]

    print(text_to_predict)
    print("Cluster:", predicted_cluster, "texts:", int(texts_per_cluster[predicted_cluster])),
    for term in ordered_words[predicted_cluster, :10]:
        print("\t" + words[term])
    paralist = []
    k = 0
    for i in labels:
        if (i == predicted_cluster):
            paralist.append(k)
        k += 1
    return paralist


cluster = Cluster(doclist, cdict, ddict, tfidf[0], tfidf[1])


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

        high = [synset.path_similarity(ss) for ss in synsets2 if synset.path_similarity(ss) != None]
        best_score = 0

        if len(high) != 0:
            best_score = max(high)

        if best_score is not None:
            score += best_score
            count += 1

    if count > 0:
        score /= count
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
    print('id is', doc_id)
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
            "docchoice": doc_id,
            "query": data['name'].lower(),
            "parachoice": data['rel_para'],
            "output": output2
        }
        user.insert_one(emp_rec1)

    user.update_many(
        {"name": data['username'], "query": data['name'].lower()},
        {
            "$set": {
                "docchoice": doc_id,
                "parachoice": data['rel_para'],
                "output": output2
            },
            "$currentDate": {"lastModified": True}

        }
    )
    return jsonify({"success": True})

global flag


@app.route('/input/<query>', methods=['POST'])
def get_all_frameworks(query):
    try:
        data = request.get_json(force=True)
        print(data)
        import ast
        data = ast.literal_eval(data)
        user = mongo.db.userdata
        val = user.find_one({"name": data['username'], "query": data['query'].lower()})
        print(val)
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

            iter_n = int(query[-1])
            name = query[:-1]
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
            print(paralist)
            global doc_id

            doc_id = IDs[iter_n]
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

            output1 = []
            output1.append(paras[para])
            if para != 0:
                output1.append(paras[para2])
                output1.append(paras[para3])

            print('output1 is:', output1[0])
            return jsonify(output1)
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
            print(val['docchoice'],"pandaaram")
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
            print(paras[para],"para")
            print(paras[para2], "para2")
            print(paras[para3],"para3")

            if para != 0:
                output2.append(paras[para2])
                output2.append(paras[para3])

            choice = val['parachoice'].split(",")
            print(choice,"choice")
            pg = 0
            i = 0
            while i < len(output2):
                print(choice[i])
                if (choice[i] == "1"):
                    output1.append(output2[pg])
                    print(output2[pg])
                pg += 1
                i += 1
            return jsonify(output1)
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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
