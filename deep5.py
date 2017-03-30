import  nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle
from collections import Counter
from numpy import random

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000


def create_lecicon(pos, neg):
    lexicon =[]
    for fi in[pos,neg]:
        with open(fi,'r') as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                all_words = word_tokenize(l.lower())
                lexicon+=list(all_words)

    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)

    l2=[]
    for w in w_counts:
        if 1000>w_counts[w] > 50:
         l2.append(w)
    return l2


def sample_handeling(sample, lexicon, classification):
    featureset=[]

    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            curent_words = word_tokenize(l.lower())
            curent_words=[lemmatizer.lemmatize(i) for i in curent_words]
            features = np.zeros(len(lexicon))
            for word in curent_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value]+=1

            features = list(features)
            featureset.append([features, classification])

    return featureset

def create_feature_sets_and_labels(pos, neg, test_size =0.1)
    lexicon = create_lecicon(pos, neg)
    featues =[]
    featues+=sample_handeling('pos.txt',lexicon,[1,0])
    featues+=sample_handeling('neg.txt',lexicon,[0,1])
    random.shuffle(featues)

    featues=np.array(featues)

    test_size=int(test_size*len(featues))
    train_x =list(featues[:,0])






