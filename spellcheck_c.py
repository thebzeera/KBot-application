import re
import time
import pickle


start_time = time.time()

def edit_distance(str1, str2):
    str1 = ' ' + str1
    str2 = ' ' + str2
    d = {}
    x = len(str1)
    y = len(str2)

    for i in range (x):
        d[i,0] = i
    for j in range (y):
        d[0,j] = j
    for j in range(1,y):
        for i in range(1,x):
            if str1[i] == str2[j]:
                d[i, j] = d[i-1, j-1]
            else:
                d[i, j] = min(d[i-1, j], d[i, j-1], d[i-1, j-1]) + 1;
    return d[x-1, y-1]

# = open("test_words.txt",'rb')
#words_list = pickle.load(words)
with open('test_words.txt', 'r') as f:
    words_list = [line.strip() for line in f]

def min_edit_distance(keywords):
    print(words_list)
    corr_query=[]
    keywords_list=keywords.split()
    for query in keywords_list:

        query = query.lower()
        my_dict = {}
        for i in range (len(words_list)):
            result = edit_distance(query, words_list[i])
            my_dict[words_list[i]] = result
        
        corr_query.append(str(min(my_dict, key=my_dict.get)))


    if corr_query==keywords_list:
        return 1
    else:
        r = keywords
        for i in range(len(corr_query)):
            r= r.replace(keywords_list[i],corr_query[i])
        print('corrected query ', r)
        return r

# if __name__ =='__main__':
#     print(corr_query)
    
