import pickle
import re
from collections import Counter
import os


# def word(file):
#     """
#
#         :param file: path of the dataset directory
#         :return: JSON file
#         """
#     print("something")
#     global full_text
#     full_text=[]
#
#     file_path = file
#     print(file_path)
#     for i in file_path:
#         add_slash = os.path.join(i, '', '')
#         for filename in os.listdir(i):
#             file_path = add_slash + filename
#             if os.path.isfile(file_path):
#                 with open(file_path, 'r', encoding='utf-8') as text:
#                     print(file_path)
#                     text = text.read()
#                     full_text.append(text)
#     file = open('clusters/spell_check.sav', 'wb')
#
#
#     return pickle.dump(full_text, file)

file = open('clusters/spell_check.sav', 'rb')
    # print(file)
    # global text
text = pickle.load(file)
def words(text):
    # print("bgdh")

    return re.findall(r'\w+', text.lower())
    # print(text)
WORDS = Counter(words(text))

def P(word, N=sum(WORDS.values())):
    "Probability of `word`."
    return WORDS[word] / N
#
def correction(word):
    "Most probable spelling correction for word."
    # return max(candidates(word), key=P)
    return max(candidates(word), key=P)
#
def candidates(word):
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


# v=candidates("intensie")
# print("Did you mean",v)