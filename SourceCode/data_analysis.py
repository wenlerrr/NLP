import json
import matplotlib.pyplot as plt
import nltk
from nltk.stem import PorterStemmer
import random
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import multiprocessing
import sys
import string
import math
from collections import Counter
import pandas as pd
import copy
import time
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
ps = PorterStemmer()  # This is for nltk stemming

stop_words_custom = ["'ve","'s","n't","...","\"\"","``","''","'m"]
jjList = ['JJ', 'JJR', 'JJS']

do_not_stem_list = ["the","this","was","very",'really']
my_stem_ref = {'got':'get',
'gotten':'get',
'getting':'get',
'try':'try',
'tried':'try',
'tries':'try',
'trying':'try',
}

def check_corpus(corpus_name):
    """
    This function checks whether a corpus is available for referencing during Sentence Segmentation,
    Tokenization and POS Tagging. If not available, it will automatically download.
    :param corpus_name: name of the corpus
    """
    try:
        nltk.data.find('corpus_name')
        print("{} is installed".format(corpus_name))
    except LookupError:
        print("{} is not installed. Trying to download now...".format(corpus_name))
        nltk.download(corpus_name)


def my_stem(word):
    """
    This function perform stemming
    :param word: this non-stemmed word
    :return: stemmed word
    """
    if word in do_not_stem_list:
        return word
    if word in my_stem_ref:
        return my_stem_ref[word]
    else:
        return ps.stem(word)


def find_most_frequent(data_list, key, top_n):
    """
    This function receives a list of json objects with keys and corresponding values.
    It counts the number of json objects with identical value of the specified key and sort the result in a list.
    Ex:
    :param data_list: a list of json objects
    :param key: the key corresponding to value we want to count
    :param top_n: only return top_n result (if None it will return all the result)
    :return: a list of tuple containing (value, count)
    """
    attr_list = [item[key] for item in data_list]
    result = Counter(attr_list).most_common()
    if top_n is None:
        return result
    if len(result) > top_n:
        return [result[i] for i in range(top_n)]
    return result

def count_freq(data_list, key):
    attr_list = [item[key] for item in data_list]
    result = Counter(attr_list)
    return list(result.items())

def plot_frequency(result,
                   title,
                   x_label,
                   y_label,
                   bar_direction):
    """
    Plot bar graphs
    :param result: a list of tuple from "find_most_frequent" function
    :param title: title of the graph
    :param x_label: x-axis legend
    :param y_label: y-axis legend
    :param bar_direction: "v" for vertical or "h" for horizontal
    """
    # setup the plot
    fig, ax = plt.subplots()
    plt.title(title)
    y = [item[1] for item in result]

    x = [item[0] for item in result]
    if bar_direction == "v":
        y.reverse()
        x.reverse()
        ax.barh(x, y, color="blue")
        for i, v in enumerate(y):
            ax.text(v, i, " " + str(v), color='blue', va='center', fontweight='bold')
        plt.subplots_adjust(left=0.3)
    else:
        ax.bar(x, y, color="blue")
        for i, v in enumerate(y):
            ax.text(i, v, str(v) + "\n", color='blue', va='center', fontweight='bold')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig('./output/{}.png'.format(title))

def plot_all_frequency(result,
                   title,
                   x_label,
                   y_label,
                   bar_direction,tick_freq):
    """
    Plot bar graphs
    :param result: a list of tuple from "find_most_frequent" function
    :param title: title of the graph
    :param x_label: x-axis legend
    :param y_label: y-axis legend
    :param bar_direction: "v" for vertical or "h" for horizontal
    """
    # setup the plot
    fig, ax = plt.subplots()
    plt.title(title)
    y = [item[1] for item in result]

    x = [item[0] for item in result]
    if bar_direction == "v":
        y.reverse()
        x.reverse()
        ax.barh(x, y, color="blue")
        for i, v in enumerate(y):
            ax.text(v, i, " " + str(v), color='blue', va='center', fontweight='bold')
        plt.subplots_adjust(left=0.3)
    else:
        ax.bar(x, y, color="blue")
        
    plt.xlabel(x_label)
    # plt.xticks(np.arange(min(list(map(int,x))), max(list(map(int, x)))+1, 5.0))
    plt.xticks(np.arange(0, max(list(map(int, x)))+1, int(tick_freq)))

    plt.ylabel(y_label)
    plt.savefig('./output/{}.png'.format(title))

def remove_stop_words(counted_list):
    """
    Remove stop words
    :param counted_list: A list of tuple containing (word, count)
    :return: the counted_list with stop words removed
    """
    cleaned_counted_list = [item for item in counted_list if not (item[0].lower() in stop_words_list)]
    return cleaned_counted_list


def process_review(text):
    review = json.loads(text)
    # remove unwanted attr
    del review['date']
    del review['review_id']

    sentence_tokenize_list = sent_tokenize(review["text"])
    # Store result back into review
    review["sentence_tokenize"] = sentence_tokenize_list
    review["num_sentences"] = str(len(sentence_tokenize_list))
    # Word Tokenize and Stemming
    stemmed_words = []
    non_stemmed_words = []
    pos=[]
    neg=[]
    neu=[]
    posTagList = []
    for sentence in sentence_tokenize_list:
        word_list = word_tokenize(sentence)
        non_stemmed_words += word_list
        for word in word_list:
            stemmed_words.append(my_stem(word.lower()))
        posTagList.append(nltk.pos_tag(word_list))

    if(review['stars']>3.0):
        pos+=stemmed_words
    elif(review['stars']<3.0):
        neg+=stemmed_words
    else:
        neu+=stemmed_words

    review['pos']=pos
    review['neg']=neg
    review['neu']=neu
    review['posTag'] = posTagList
    review["non_stemmed_words"] = non_stemmed_words
    review["num_non_stemmed_words"] = str(len(set(non_stemmed_words)))  # Set is needed (unrepeated count)
    review["stemmed_words"] = stemmed_words
    review["num_stemmed_words"] = str(len(set(stemmed_words)))  # Set is needed (unrepeated count)

    return review


def log_result(retrieval):
    results.append(retrieval)
    all_non_stemmed_words.extend(retrieval["non_stemmed_words"])
    all_stemmed_words.extend(retrieval["stemmed_words"])
    all_pos.extend(retrieval["pos"])
    all_neg.extend(retrieval["neg"])
    all_neu.extend(retrieval["neu"])
    sys.stderr.write('\rdone {0:%}'.format(len(results) / n_reviews))

#Most Frequent Adjectives for each Rating.
def countAdj(Adjs):
    counts = Counter(x[0].lower() for x in Adjs if x[1] in jjList)
    counts_dict = dict(counts.most_common(10))
    return counts_dict

def countObs(_StarReview):
    _starDict = {}
    for review in _StarReview:
        set_of_adj = set()
        for sentence in review:
            for x in sentence:
                if x[1] in jjList:
                    set_of_adj.add(x[0].lower())
        for item in set_of_adj:
            _starDict[item] = _starDict.get(item, 0) + 1
    return _starDict

def computeIndicativeness(allReviewsList, reviewListAdj):
    nWords = sum(allReviewsList.values())

    for i in range (1,6):
        indicative_dict = {}
        nWordsRating = sum(reviewListAdj[i - 1].values())
        for word in reviewListAdj[i - 1]:
            pw_r = reviewListAdj[i - 1][word] / nWordsRating
            pw = allReviewsList[word] / nWords
            indicative_dict[word] = pw_r * math.log(pw_r / pw)
        print("\n===========================================================")
        print('Top 10 Indicative Adjectives for ',i,'star review by')
        print(Counter(indicative_dict).most_common(10))
        print("===========================================================\n")

def nlpApplication(all_pos,all_neg,all_neu):
    print("\nPlease wait...\n")
    # Sentiment Analysis
    all_pos_words = all_pos
    all_neg_words = all_neg
    all_neu_words = all_neu

    allPosLower = [item.lower() for item in all_pos_words]
    allNegLower = [item.lower() for item in all_neg_words]
    allNeuLower = [item.lower() for item in all_neu_words]
    sid = SentimentIntensityAnalyzer()
    pos_word_list = []
    neu_word_list = []
    neg_word_list = []

    for word in all_stemmed_words:
        word = word.lower()
        if (sid.polarity_scores(word)['compound'] >= 0.4) and (word in allPosLower):
            pos_word_list.append(word)
        elif (sid.polarity_scores(word)['compound']) <= -0.4 and (word in allNegLower):
            neg_word_list.append(word)
        else:
            neu_word_list.append(word)

    pos_frequency = Counter(pos_word_list).most_common()
    neu_frequency = Counter(neu_word_list).most_common()
    neg_frequency = Counter(neg_word_list).most_common()

    pos_frequency = remove_stop_words(pos_frequency)
    neu_frequency = remove_stop_words(neu_frequency)
    neg_frequency = remove_stop_words(neg_frequency)

    print("\n=====+====NLP APPLICATION: Sentiment analysis=============")
    print("\nPositive words frequency")
    print(len(pos_frequency))
    for i in range(10):
        print(pos_frequency[i])

    print("\nNeutral words frequency")
    print(len(neu_frequency))
    for i in range(10):
        print(neu_frequency[i])
    print("\nNegative words frequency")
    print(len(neg_frequency))
    for i in range(10):
        print(neg_frequency[i])

    sentence = input("Please input a sentence:")
    sentiment_analyzer_scores(sentence)

def sentiment_analyzer_scores(sentence):
    analyser = SentimentIntensityAnalyzer()
    score = analyser.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(score)))
    print("===========================================================\n")

def plotGraph(top_10_num_sentences,sentence_freq, results):

    #Plotting results

    for i in range(5):
        plot_all_frequency(sorted(sentence_freq[i], key=lambda tup: int(tup[0])),
                    "Distribution of number of sentences for " + str(i+1) +" star",
                    "No. of sentences",
                       "No. of reviews",
                       "h",5)

        plot_frequency(top_10_num_sentences[i],
                       "Top 10 Number Sentences for " + str(i+1) +" star",
                       "No. of sentences",
                       "No. of reviews",
                       "h")

    top_10_num_non_stemmed_words = find_most_frequent(results, "num_non_stemmed_words", 10)
    plot_frequency(top_10_num_non_stemmed_words,
                   "Top 10 length of review ( before stemming)",
                   "No. of tokens",
                   "No. of reviews",
                   "h")

    top_10_num_stemmed_words = find_most_frequent(results, "num_stemmed_words", 10)
    plot_frequency(top_10_num_stemmed_words,
                   "Top 10 length of review ( after stemming)",
                   "No. of tokens",
                   "No. of reviews",
                   "h")

    review_length_distribution_nonstem=list(count_freq(results, "num_non_stemmed_words"))
    plot_all_frequency(sorted(review_length_distribution_nonstem, key=lambda tup: int(tup[0])),
                   "Distribution of length of review (before stemming)",
                   "No. of tokens",
                   "No. of reviews",
                   "h",20)

    review_length_distribution_stem=list(count_freq(results, "num_stemmed_words"))
    plot_all_frequency(sorted(review_length_distribution_stem, key=lambda tup: int(tup[0])),
                   "Distribution of length of review (after stemming)",
                   "No. of tokens",
                   "No. of reviews",
                   "h",20)
    plt.show()

if __name__ == "__main__":
    # Load necessary corpus
    for corpus in ["stopwords", "punkt", "averaged_perceptron_tagger"]:
        check_corpus(corpus)
    stop_words_list = stopwords.words("english") + list(string.punctuation) + stop_words_custom

    data = []
    # file_name = './raw/reviewSamples20.json'
    file_name = './raw/reviewSelected100.json'
    with open(file_name) as file:
        for line in file:
            data.append(line)
    # Sentence Segmentation
    n_reviews = len(data)

    # Declare golbal variable to store result
    results = []
    # Variable for word tokenization
    all_non_stemmed_words = []
    all_stemmed_words = []
    all_pos=[]
    all_neg=[]
    all_neu=[]


    # perform parallel processing (must faster than serial)
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
    for item in data:
        pool.apply_async(process_review, args=[item], callback=log_result)
    pool.close()
    pool.join()
    print("")  # print linebreak


    # Print 5 sample sentences (result from sentence segmentation)
    random.seed(2004)  # seed to make random the same for consistency
    n_samples = 5
    print("\n===========================================================")
    print("{} sampled sentences".format(n_samples))
    i = 0
    while i < 5:
        sample_review = random.choice(results)
        sample_sentence_list = sample_review["sentence_tokenize"]
        sample_sentence = random.choice(sample_sentence_list)
        if sample_sentence not in stop_words_list:  # sometimes sentence segmentation break "!" into a sentence
            i += 1
            print(sample_sentence)
    print("===========================================================\n")

    # Print 5 sample reviews
    random.seed(2004)  # seed to make random the same for consistency
    n_samples = 5
    print("\n===========================================================")
    print("{} sampled reviews".format(n_samples))
    i = 0
    while i < 5:
        sample_review = random.choice(results)
        print("Review Sample ",str(i+1),": \n" ,sample_review['text'],"\n")
        sample_sentence_list = sample_review["sentence_tokenize"]
        for sentence in sample_sentence_list:
            print(sentence)
        i +=1
        print("-------------------------")
    print("===========================================================\n")

    # Count words
    non_stemmed_words_frequency = Counter(all_non_stemmed_words).most_common()
    stemmed_words_frequency = Counter(all_stemmed_words).most_common()

    # Remove Stop Words (defined on top)
    non_stemmed_words_frequency = remove_stop_words(non_stemmed_words_frequency)
    stemmed_words_frequency = remove_stop_words(stemmed_words_frequency)


    print("\n===========================================================")
    print("List the top-20 Non stemmed most frequent words")
    print(Counter(non_stemmed_words_frequency).most_common(20))

    print("\nList the top-20 stemmed most frequent words")
    print(Counter(stemmed_words_frequency).most_common(20))
    print("===========================================================\n")

    # POS Tagging
    n_samples = 5
    print("\n===========================================================")
    print("{} POS Tagging".format(n_samples))
    for i in range(5):
        sample_review = random.choice(results)
        sample_sentence_list = sample_review["sentence_tokenize"]
        sample_sentence = random.choice(sample_sentence_list)
        print(sample_sentence)
        word_tokens = word_tokenize(sample_sentence)
        print(nltk.pos_tag(word_tokens),"\n")
    print("===========================================================\n")


    #Partition Results into Stars
    df = pd.DataFrame(results)
    top_10_num_sentences=[]
    top_10_num_sentences_freqAdj = []
    AdjDict = {}
    AdjList = []
    countAppearance = {}
    reviewListAdj = []
    allReviewsList = Counter()
    sentence_freq=[]

    for i in range(1,6):
        ratingsDF =  df[df['stars'] == i]
        top_10_num_sentences.append(find_most_frequent(ratingsDF.to_dict(orient='records'), "num_sentences", 10))
        sentence_freq.append(list(count_freq(ratingsDF.to_dict(orient='records'), "num_sentences")))

        l = list(df[df['stars'] == i]['posTag'])
        _StarReview = [subList for subList in l ]
        AdjList.append([item2 for subList in l for item in subList for item2 in item])

        # Count Rating for each Rating
        counts_dict = countAdj(AdjList[i-1])
        #Top 10 most frequent Adj for each Rating
        print("\n===========================================================")
        print(str(i),' star: Top 10 most frequent Adjectives for each Rating')
        print(Counter(counts_dict).most_common(10))
        print("===========================================================\n")

        _starDict = countObs(_StarReview)
        reviewListAdj.append(_starDict)
        allReviewsList += Counter(_starDict)

    #Copute Indicativness of Adj
    computeIndicativeness(allReviewsList, reviewListAdj)


    #NLP Application
    nlpApplication(all_pos, all_neg, all_neu)

    choice = int(input('Show Graphs?:\n1. Yes\n2. No\nEnter Choice:'))
    if choice ==1:
        #Plot Graphs for top 10 non-stemmed and stemmed words
        plotGraph(top_10_num_sentences, sentence_freq,results)
     




    
