import pandas as pd
import re
import nltk
import numpy as np


def remove_non_ascii(s):
    ''' Removes all non-ascii characters.
    '''
    return ''.join(i for i in s if ord(i) < 128)


def clean_text(t):
    ''' Cleans up a text body.
    '''
    t = re.sub('[\n\t\r]', ' ', t)
    t = re.sub(' +', ' ', t)
    t = re.sub('<.*?>', '', t)
    t = remove_non_ascii(t)
    t = t.lower()
    t = re.sub(r"what's", "what is ", t)
    t = t.replace('(ap)', '')
    t = re.sub(r"\'ve", " have ", t)
    t = re.sub(r"can't", "cannot ", t)
    t = re.sub(r"n't", " not ", t)
    t = re.sub(r"i'm", "i am ", t)
    t = re.sub(r"\'s", "", t)
    t = re.sub(r"\'re", " are ", t)
    t = re.sub(r"\'d", " would ", t)
    t = re.sub(r"\'ll", " will ", t)
    t = re.sub(r'\s+', ' ', t)
    t = re.sub(r"\\", "", t)
    t = re.sub(r"\'", "", t)
    t = re.sub(r"\"", "", t)
    t = re.sub(r'\W+', ' ', t)
    t = remove_non_ascii(t)
    t = t.strip()
    return t


def text_preprocessing(text):
    """
    Cleaning and parsing the text.
    """
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    nopunc = clean_text(text)
    tokenized_text = tokenizer.tokenize(nopunc)
    # remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]
    combined_text = ' '.join(tokenized_text)
    return combined_text


def read_data(filename: str) -> pd.DataFrame:
    data = pd.read_csv(filename, sep='\t', header=None)
    data.columns = ['polarity', 'aspect_category', 'target_term', 'character_offsets', 'sentence']
    return data


def label_to_cat(label):
    if label == 'positive':
        return 2
    elif label == 'neutral':
        return 1
    elif label == 'negative':
        return 0
    else:
        raise Exception()


def cat_to_label(cat):
    if cat == 2:
        return 'positive'
    elif cat == 1:
        return 'neutral'
    elif cat == 0:
        return 'negative'
    else:
        raise Exception()


def process_data(filename: str) -> pd.DataFrame:
    data = read_data(filename)
    data['clean_sentence'] = data['sentence'].apply(lambda x: text_preprocessing(x))
    data['clean_target_term'] = data['target_term'].apply(lambda x: text_preprocessing(x))
    data['label'] = data['polarity'].apply(lambda x: label_to_cat(x))
    return data


def pad_and_truncate(sequence, maxlen, dtype='int64', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    x[:len(trunc)] = trunc
    return x


def mask_pad(sequence):
    sequence[sequence != 0] = 1
    return sequence
