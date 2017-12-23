import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from scipy.stats import spearmanr

import re

import nltk
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))

def evaluate(y_true, y_pred):
    prediction = np.array(y_pred)
    answer = np.array(y_true)
    
    bigger_than_05_indices = answer>0.5
    prediction_05 = prediction[bigger_than_05_indices]
    answer_05 = answer[bigger_than_05_indices]
    
    pearson_correlation = pearsonr(prediction, answer)[0]
    spearman_correlation = spearmanr(prediction,answer)[0]
    
    pearson_correlation_05 = pearsonr(prediction_05, answer_05)[0]
    spearman_correlation_05 = spearmanr(prediction_05,answer_05)[0]
    
    return (pearson_correlation, spearman_correlation, pearson_correlation_05, spearman_correlation_05)

def pearson_score(y_true, y_pred,*args):
    prediction = np.array(y_pred)
    answer = np.array(y_true)
    
    return pearsonr(prediction, answer)[0]

def get_emotion_data(emotion_name, column_names = ['id','text','emotion','intensity']):
    train_data = pd.read_csv('source_data/train/{0}-ratings-0to1.train.txt'.format(emotion_name),
                             delimiter='\t',header=None, names = column_names)
    dev_data = pd.read_csv('source_data/dev/{0}-ratings-0to1.dev.gold.txt'.format(emotion_name),
                           delimiter='\t',header=None, names = column_names)
    test_data = pd.read_csv('source_data/test/{0}-ratings-0to1.test.gold.txt'.format(emotion_name),
                            delimiter='\t',header=None, names = column_names)
    
    return train_data, dev_data, test_data


def text_to_wordlist(text, remove_stopwords=False, w2v=None):
    text = re.sub(r"@\w{1,15}", "USERNAME", text)
    text = re.sub(r"[^A-Za-z0-9^,!.#\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " !", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    text = text.lower().split()
    
    if remove_stopwords:
        text = [w for w in text if not w in stops]
    
    if w2v is not None:
        text = ['#hashtag' if ((w not in w2v) and '#' in w) else w for w in text]
        
    text = " ".join(text)
    
    #text = "".join([' {0} '.format(main_emojis[c]) if c in main_emojis else c for c in text])

    
    return(text)