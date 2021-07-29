import unicodedata
import re
import json
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords


import pandas as pd
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.model_selection import train_test_split


def basic_clean(text):
    ''' 
    This function takes in the original text.
    The text is all lowercased and any characters that are not ascii are ignored
    additionally, special characters are all removed.
    A clean verion of the text is then returned
    '''
    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii','ignore').decode('utf-8','ignore')
    text = re.sub(r"[^a-z0-9'\s]",'',text)
    return text


def tokenize(text):
        '''
    This function takes in text
    and returns the text as individual tokens put back into the original text
    '''
    tokenizer = nltk.tokenize.ToktokTokenizer()
    text = tokenizer.tokenize(text, return_str=True)
    return text


def stem(text):
      '''
    This function takes in text
    and returns the stem word joined back into the original text
    '''
    ps = nltk.porter.PorterStemmer()
    stems = [ps.stem(word) for word in text.split()]
    text_stemmed = ' '.join(stems)
    return text_stemmed


def lemmatize(text):
    '''
    This function takes in text
    and returns the lemmatized word joined back into the text
    '''
    wnl = nltk.stem.WordNetLemmatizer()
    lemmas = [wnl.lemmatize(word) for word in text.split()]
    text_lemmatized = ' '.join(lemmas)
    return text_lemmatized


def remove_stopwords(text, extra_words=[], exclude_words=[]):
    '''
    This function takes in text, extra words and exclude words
    and returns a list of text with stopword removed
    '''
    #nltk.download('stopwords')
    stopword_list = stopwords.words('english')
    if len(extra_words) > 0 :
        for word in extra_words:
            stopword_list.append(word)
    if len(exclude_words) > 0:
        for word in exclude_words:
            stopword_list.remove(word)
    words = text.split()
    filtered_words = [word for word in words if word not in stopword_list]
    text_without_stopwords = ' '.join(filtered_words)
    return text_without_stopwords   


def prep_article_data(df, column, extra_words=[], exclude_words=[]):
    '''
    This function take in a df and the string name for a text column with 
    option to pass lists for extra_words and exclude_words and
    returns a df with the text article title, original text, stemmed text,
    lemmatized text, cleaned, tokenized, & lemmatized text with stopwords removed.
    '''
    df['clean'] = df[column].apply(basic_clean)\
                            .apply(tokenize)\
                            .apply(remove_stopwords, 
                                   extra_words=extra_words, 
                                   exclude_words=exclude_words)
    
    df['stemmed'] = df[column].apply(basic_clean)\
                            .apply(tokenize)\
                            .apply(stem)\
                            .apply(remove_stopwords, 
                                   extra_words=extra_words, 
                                   exclude_words=exclude_words)
    
    df['lemmatized'] = df[column].apply(basic_clean)\
                            .apply(tokenize)\
                            .apply(lemmatize)\
                            .apply(remove_stopwords, 
                                   extra_words=extra_words, 
                                   exclude_words=exclude_words)
    
    return df[['title', column,'clean', 'stemmed', 'lemmatized']]



def split(df, stratify_by=None):
   
    train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[stratify_by])
    
    train, validate = train_test_split(train, test_size=.3, random_state=123, stratify=train[stratify_by])
    
    return train, validate, test