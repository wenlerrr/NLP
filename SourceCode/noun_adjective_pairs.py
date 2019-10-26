import spacy
from collections import Counter
import json
import spacy
nlp = spacy.load('en_core_web_sm')
from itertools import groupby
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk import Tree
from nltk import pos_tag
from nltk.chunk import RegexpParser
from nltk import ne_chunk
import copy 

#Defining functions
def check_corpus(corpus_name):
    """
    This function checks whether a corpus in nltk is installed. If not available, it will automatically download.
    :param corpus_name: name of the corpus
    """
    try:
        nltk.data.find('corpus_name')
        print("{} is installed".format(corpus_name))
    except LookupError:
        print("{} is not installed. Trying to download now...".format(corpus_name))
        nltk.download(corpus_name)

def find_most_frequent(data_list, key, top_n):
    attr_list = [item[key] for item in data_list]
    result = Counter(attr_list).most_common()
    if top_n is None:
        return result
    if len(result) > top_n:
        return [result[i] for i in range(top_n)]
    return result

def tokenizer_master(sentence):
    sentss = sent_tokenize(sentence.lower())
    sentss2 = []
    for each_sent in sentss:
        sentss2.extend([word_tokenize(each_sent)])
    return sentss2

def tagged_sents(document):
    tagged_sentences = [pos_tag(sent) for sent in document]
    return tagged_sentences

def get_chunks(tagged_sentences):
    master_list = []
    master_noun = []
    master_adj = []
    grammar = r"""
    CHUNK1:
        {<NN.*><.*>?<JJ.*>}  # Any Noun terminated with Any Adjective
    
    CHUNK2:
        {<NN.*|JJ.*><.*>?<NN.*>}  # Nouns or Adjectives, terminated with Nouns
    """
    cp = RegexpParser(grammar)
    for sent in tagged_sentences:
        tree = cp.parse(sent)
        for subtree in tree.subtrees(filter = lambda t: t.label() in ['CHUNK1', 'CHUNK2']):
            if (str(subtree).find('NN') >0 or str(subtree).find('NNS') >0 or str(subtree).find('NNP') >0) and (str(subtree).find('JJ')>0 or str(subtree).find('JJS')>0 or str(subtree).find('JJR')>0):
                nouns = [word for word, tag in subtree.leaves() if tag in ['NN', 'NNS','NNP']]
                adjss = [word for word, tag in subtree.leaves() if tag in ['JJ','JJR','JJS']]
                master_noun.extend([nouns])
                master_adj.extend([adjss])
    return [m[0]+":"+n[0] for m,n in zip(master_noun,master_adj)]

def sortBizNounAdjFreq(business):
    sorted_business = sorted(business.items(), key=lambda kv: kv[1], reverse=True)
    return sorted_business

def mergeCommon(arrayCountDict):
    for key in arrayCountDict:
        tempList1 = []
        naPairs = set()
        for j in range(len(arrayCountDict[key])):
            count = copy.deepcopy(arrayCountDict[key][j][1])
            ll = arrayCountDict[key][j][0].split(":")
            tempTuple = (ll[0], ll[1])
            invertedTempTuple = (ll[1], ll[0])
            if(invertedTempTuple in naPairs):
                continue
            else:
                naPairs.add(tempTuple)
                tempList1.append((tempTuple[0]+":"+tempTuple[1], count))
        arrayCountDict[key] = tempList1
    return arrayCountDict

#Main body of code
for corpus in ["stopwords", "punkt", "averaged_perceptron_tagger"]:
        check_corpus(corpus)

data = []
file_name = './raw/reviewSelected100.json'
with open(file_name) as file:
    for line in file:
        data.append(line)
n_reviews = len(data)

for i in range(len(data)):
    data[i] = json.loads(data[i])

top_5_business_id = find_most_frequent(data, "business_id", 5)

business_reviews = {}
businessIDs = set()
business_noun_adjective_pairs = {}

for i in range(len(top_5_business_id)):
    business_reviews[top_5_business_id[i][0]] = []
    business_noun_adjective_pairs[top_5_business_id[i][0]] = []
    businessIDs.add(top_5_business_id[i][0])

for i in range(len(data)):
    if(data[i]['business_id'] in businessIDs):
        business_reviews[data[i]['business_id']].append(data[i]['text'])

pairDict = {}
for i in range(len(top_5_business_id)):
    pairDict[top_5_business_id[i][0]] = []

for key in business_reviews:
    for j in range(len(business_reviews[key])):
        tempList = get_chunks(tagged_sents(tokenizer_master(business_reviews[key][j])))
        pairDict[key].extend(tempList)

countDict = {}
for i in range(len(top_5_business_id)):
    countDict[top_5_business_id[i][0]] = {}

for key in business_reviews:
    for i in range(len(pairDict[key])):
        if(pairDict[key][i] in countDict[key]):
            continue
        else:
            countDict[key][pairDict[key][i]] = pairDict[key].count(pairDict[key][i])

arrayCountDict = {}
for key in countDict:
    arrayCountDict[key] = sortBizNounAdjFreq(countDict[key])

arrayCountDict = mergeCommon(arrayCountDict)

for key in arrayCountDict:
    print("Noun-adjective pairs for " + key + ":")
    print(arrayCountDict[key])
#ZBE-H_aUlicix_9vUGQPIQ: food:good(7), food:great(6), place:great(5),food:average(4),service:great(4)
#e-YnECeZNt8ngm0tu4X9mQ: food:good(8),place:good(4),service:great(4),bbq:authentic(4),staff:attentive(3)
#j7HO1YeMQGYo3KibMXZ5vg: food:good(8), portions:huge(6), bit:little(4), service:good(4), service:great(4)
#7e3PZzUpG5FYOTGt3O3ePA: service:excellent(7), food:great(5), price:reasonable(4), food:good(4), service:great(4)
#vuHzLZ7nAeT-EiecOkS5Og: service:great(7), service:crappy(5), job:great(4), reviews:negative(4), brand:new(4)



