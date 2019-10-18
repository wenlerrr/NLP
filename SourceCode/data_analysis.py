import json
from collections import Counter
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
import pickle
from collections import Counter
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
ps = PorterStemmer()  # This is for nltk stemming
#stop_words_custom = ['n\'t', 'one', 'two', '\'s', 'would', 'get', 'very','...']
stop_words_custom = []
jjList = ['JJ', 'JJR', 'JJS']

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


#do_not_stem_list = ["the", "this", "was", "battery", "charge", "because", "very", "verify"]
do_not_stem_list = []
# my_stem_ref = {
#     "charged": "charge",
#     "charges": "charge",
#     "charger": "charge",
#     "charging": "charge",
#     "batteries": "battery",
#     "batterys": "battery",
#     "verified": "verify",
#     "verifies": "verify"
# }
my_stem_ref = {}

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
    my_list = [{"name": "John"}, {"name": "Marry"}, {"name": "John"}]
    result = find_most_frequent(my_list, "name", None)
    >> result = [("John", 2), ("Marry", 1)]
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

    # {"review_id": "fdiNeiN_hoCxCMy2wTRW9g", "user_id": "w31MKYsNFMrjhWxxAb5wIw",
    #  "business_id": "eU_713ec6fTGNO4BegRaww", "stars": 4.0, "useful": 0, "funny": 0, "cool": 0,
    #  "text": "I'll be the first to admit that I was not excited about going to La Tavolta. Being a food snob, when a group of friends suggested we go for dinner I looked online at the menu and to me there was nothing special and it seemed overpriced.  Im also not big on ordering pasta when I go out. Alas, I was outnumbered. Thank goodness! I ordered the sea bass special. It was to die for. Cooked perfectly, seasoned perfectly, perfect portion. I can not say enough good things about this dish. When the server asked how it was he seemed very proud of the dish and said, \" doesn't she (the chef) do an incredible job?\" She does. \n\nMy hubby got the crab tortellini and also loved his. I heard \"mmmm this is so good\" from all around the table. Our waiter was super nice and even gave us free desserts because we were some of the last people in the restaurant. Service was very slow and the place was PACKED but we had our jugs of wine and a large group with good conversation so it didn't seem to bother anyone.\n\nSo-\n\nDo order the calamari and fried zucchini appetizers. Leave out the mussels. \n\nIf they have the sea bass special, I highly recommend it. The chicken parm and crab tortellini were also very good and very big. The chicken Romano was a bit bland. The house salads were teeny. \n\nDo make a reservation but still expect to wait for your food. Go with a large group of people and plan for it to be loud. Don't go with a date unless you're fighting and don't feel like hearing anything they have to say.  Ask to sit in the side room if it's available.",
    #  "date": "2013-01-20 13:25:59"}

    sentence_tokenize_list = sent_tokenize(review["text"])
    del review['text']
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
    counts = Counter(x[0] for x in Adjs if x[1] in jjList)
    counts_dict = dict(counts.most_common(10))
    return counts_dict

def countObs(_StarReview):
    _starDict = {}
    for review in _StarReview:
        set_of_adj = set()
        for sentence in review:
            for x in sentence:
                if x[1] in jjList:
                    set_of_adj.add(x[0])
        for item in set_of_adj:
            _starDict[item] = _starDict.get(item, 0) + 1
    return _starDict

def computeIndicativeness(allReviewsList, reviewListAdj):
    indicativeProp = []
    nWords = sum(allReviewsList.values())

    for i in range (1,6):
        indicative_dict = {}
        nWordsRating = sum(reviewListAdj[i - 1].values())
        for word in reviewListAdj[i - 1]:
            pw_r = reviewListAdj[i - 1][word] / nWordsRating
            pw = allReviewsList[word] / nWords
            indicative_dict[word] = pw_r * math.log(pw_r / pw)
        indicativeProp.append(indicative_dict)
        print('top 10 adj for ',i,'star review are: ', Counter(indicative_dict).most_common(10))


if __name__ == "__main__":
    # Load necessary corpus
    for corpus in ["stopwords", "punkt", "averaged_perceptron_tagger"]:
        check_corpus(corpus)
    stop_words_list = stopwords.words("english") + list(string.punctuation) + stop_words_custom

    # Load data
    #print("Loading pickle...")
    # try:
    #     results, all_non_stemmed_words, all_stemmed_words,all_pos,all_neg,all_neu = pickle.load(open("./raw/save.p", "rb"))
    #     print("Found pickle!")
    # except FileNotFoundError:
    #print("Data is not processed yet. Processing now...")

    data = []
    #file_name = './raw/reviewSamples20.json'
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

    # save in pickle
    #pickle.dump((results, all_non_stemmed_words, all_stemmed_words,all_pos,all_neg,all_neu), open("./raw/save.p", "wb"))
    # Data set Analysis
    top_10_user_id = find_most_frequent(results, "user_id", 10)
    print('\nuser_id')
    for id, count in top_10_user_id:
        print(id + ", " + str(count))

    top_10_business_id = find_most_frequent(results, "business_id", 10)
    print('\ntop_10_business_id')
    for id, count in top_10_business_id:
        print(id + ", " + str(count))

    #Partition Results into Stars
    df = pd.DataFrame(results)
    top_10_num_sentences=[]
    top_10_num_sentences_freqAdj = []
    AdjDict = {}
    AdjList = []
    countAppearance = {}
    reviewListAdj = []
    allReviewsList = Counter()

    for i in range(1,6):
        ratingsDF =  df[df['stars'] == i]
        top_10_num_sentences.append(find_most_frequent(ratingsDF.to_dict(orient='records'), "num_sentences", 10))
        l = list(df[df['stars'] == i]['posTag'])
        # _StarReview = [item for subList in l for item in subList]
        _StarReview = [subList for subList in l ]
        AdjList.append([item2 for subList in l for item in subList for item2 in item])

        # Count Rating for 1 Star
        counts_dict = countAdj(AdjList[i-1])
        #print("COUNT ADJ with Duplicates: ",counts_dict)

        _starDict = countObs(_StarReview)
        reviewListAdj.append(_starDict)
        print(str(i)," star: ",_starDict)
        allReviewsList += Counter(_starDict)


    print("ALL STAR: " ,allReviewsList)
    countAdj([item for sublist in AdjList for item in sublist])
    computeIndicativeness(allReviewsList, reviewListAdj)


    #top_10_num_sentences.append(find_most_frequent(df[df['stars'] == i].to_dict(orient='records'), "num_sentences", 10))
    #top_10_num_sentences.append(find_most_frequent(df[df['stars'] == i].to_dict(orient='records'), "num_sentences", 10))
    top_10_num_non_stemmed_words = find_most_frequent(results, "num_non_stemmed_words", 10)
    top_10_num_stemmed_words = find_most_frequent(results, "num_stemmed_words", 10)

    # Print 5 sample sentences (result from sentence segmentation)
    random.seed(2004)  # seed to make random the same for consistency
    n_samples = 5
    print("\n{} sampled sentences".format(n_samples))
    i = 0
    while i < 5:
        sample_review = random.choice(results)
        sample_sentence_list = sample_review["sentence_tokenize"]
        sample_sentence = random.choice(sample_sentence_list)
        if sample_sentence not in stop_words_list:  # sometimes sentence segmentation break "!" into a sentence
            i += 1
            print(sample_sentence)

    # Count words
    non_stemmed_words_frequency = Counter(all_non_stemmed_words).most_common()
    stemmed_words_frequency = Counter(all_stemmed_words).most_common()

    # Remove Stop Words (defined on top)
    non_stemmed_words_frequency = remove_stop_words(non_stemmed_words_frequency)
    stemmed_words_frequency = remove_stop_words(stemmed_words_frequency)

    #Sentiment Analysis
    all_pos_words= all_pos
    all_neg_words= all_neg
    all_neu_words= all_neu

    allPosLower = [item.lower() for item in all_pos_words]
    allNegLower = [item.lower() for item in all_neg_words]
    allNeuLower = [item.lower() for item in all_neu_words]
    sid = SentimentIntensityAnalyzer()
    pos_word_list=[]
    neu_word_list=[]
    neg_word_list=[]

    for word in all_stemmed_words:
        word=word.lower()
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

    print("\nList the top-20 most frequent words")
    print("\nnon_stemmed_words_frequency")
    for i in range(10):
        print(non_stemmed_words_frequency[i])
    print("\nstemmed_words_frequency")
    for i in range(10):
        print(stemmed_words_frequency[i])

    print("\npos_words_frequency")
    print(len(pos_frequency))
    for i in range(10):
        print(pos_frequency[i])

    print("\nneu_words_frequency")
    print(len(neu_frequency))
    for i in range(10):
        print(neu_frequency[i])
    print("\nneg_words_frequency")
    print(len(neg_frequency))
    for i in range(10):
        print(neg_frequency[i])

    # POS Tagging
    n_samples = 5
    print("\n{} POS Tagging".format(n_samples))
    for i in range(5):
        sample_review = random.choice(results)
        sample_sentence_list = sample_review["sentence_tokenize"]
        sample_sentence = random.choice(sample_sentence_list)
        word_tokens = word_tokenize(sample_sentence)
        print(nltk.pos_tag(word_tokens))

    # Plotting results
    plot_frequency(top_10_user_id,
                   "Top 10 User reviewed",
                   "No. of User",
                   "User ID",
                   "v")


    for i in range(5):
        plot_frequency(top_10_num_sentences[i],
                       "Top 10 Number Sentences for " + str(i+1) +" star",
                       "No. of sentences",
                       "No. of reviews",
                       "h")


    plot_frequency(top_10_num_non_stemmed_words,
                   "Top 10 Number of Non-Stemmed Words (unrepeated)",
                   "No. of words",
                   "No. of reviews",
                   "h")

    plot_frequency(top_10_num_stemmed_words,
                   "Top 10 Number of Stemmed Words (unrepeated)",
                   "No. of words",
                   "No. of reviews",
                   "h")
    plt.show()


    
