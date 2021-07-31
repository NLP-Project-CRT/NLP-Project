#Z0096

# import standard libraries
import numpy as np
import pandas as pd
import re

# import file managers
from os.path import isfile
import pickle

# import data tools
import json
from sklearn.model_selection import train_test_split

# import language detector
from langdetect import detect

# import prepare functions
import prepare as p
from acquire import get_repo_links, scrape_github_data


#################### Pickle Data ####################


def make_pickles(py_object, filename):
    '''
    Takes in argument for python object and associated filename as 
    string and pickles object with "filename.pickle"
    '''

    # dump obtained DataFrame into pickle file
    pickle_out = open(f'{filename}.pickle', 'wb')
    pickle.dump(py_object, pickle_out)
    pickle_out.close()


def open_pickles(filename):
    '''
    Takes in filename as string of previously pickled object and
    returns opened pickle jar as a python object
    '''

    # load existing DataFrame from pickle file
    pickle_in = open(f'{filename}.pickle', 'rb')
    opened_jar = pickle.load(pickle_in)
    pickle_in.close()

    return opened_jar


#################### Wrangle Data ####################


def open_json_data():
    '''
    '''
    
    # open data file and load with json
    repos = open('data2.json')
    repos = json.load(repos)
    # remove null entries
    for i, repo in enumerate(repos):
        if repo == None:
            del(repos[i])
    # normalize nested json data into DataFrame
    df = pd.json_normalize(repos, errors='ignore')
    # drop duplicate observations and nulls
    df = df.drop_duplicates().dropna()

    return df


def get_english_only(df):
    '''
    '''
    
    # detect natural language of repo
    df['natural_language'] = df.readme_contents.apply(lambda row: detect(row))
    # filter to only Enlish results
    df = df[df.natural_language == 'en']
    
    return df


def remove_code_snippets(df):
    '''
    '''

    # remove HTML and markdown code
    df['cleaned_readme'] = df.readme_contents\
                    .apply(lambda row: re.sub(r'(\<.+\>)', '', row))
    df['cleaned_readme'] = df.cleaned_readme\
                    .apply(lambda row: re.sub(r'!?(\[.+\])(\(.+\))?', '', row))
    # remove coded spaces and page breaks
    df['cleaned_readme'] = df.cleaned_readme\
                    .apply(lambda row: re.sub(r'(\n+)', ' ', row))
    df['cleaned_readme'] = df.cleaned_readme\
                    .apply(lambda row: re.sub(r'(&nbsp;+)', '', row))
    # remove naked hyperlinks
    df['cleaned_readme'] = df.cleaned_readme\
                    .apply(lambda row: re.sub(r'http\S+', '', row))
    # replaced hyphens with spaces
    df['cleaned_readme'] = df.cleaned_readme\
                    .apply(lambda row: re.sub(r'-', ' ', row))
    
    return df


def extensive_clean(df):
    '''
    '''
    
    # run cleaner functions on text data
    df['cleaned_readme'] = df.cleaned_readme.apply(
                                    lambda row: p.remove_stopwords(
                                    p.tokenize(p.basic_clean(row)),
                                    extra_words=['&#9;']))
    
    return df


def create_char_counts(df):
    '''
    '''
    
    # create character counts for original and cleaned text
    df['original_char_length'] = df.readme_contents.apply(lambda row: len(row))
    df['cleaned_char_length'] = df.cleaned_readme.apply(lambda row: len(row))
    
    return df


def create_pct_changed(df):
    '''
    '''

    # calclate perecent change column, rounded to int
    df['pct_char_removed'] = ((df.cleaned_char_length /
                               df.original_char_length) * 100).astype(int)

    return df


def filter_language(df):
    '''
    '''
    
    # filter for programming languges with 10 or more
    df = df[df.language.isin(['Python',
                              'JavaScript',
                              'Jupyter Notebook',
                              'HTML',
                              'TypeScript',
                              'R'])]
    
    return df


def prep_github_repos():
    '''
    '''

    # load data into DataFrame
    df = open_json_data()
    # filter data to only English results
    df = get_english_only(df)
    # remove HTML, markdown, etc.
    df = remove_code_snippets(df)
    # remove non-ascii chars and stopwords, tokenize
    df = extensive_clean(df)
    # create character count columns
    df = create_char_counts(df)
    # create percent change column
    df = create_pct_changed(df)
    # filter for programming languages
    df = filter_language(df)

    return df


def polish_github_repos(df):
    '''
    '''

    # remove cleaned char count outliers
    df = df[df.cleaned_char_length.isin(
                p.filter_iqr_outliers(df.cleaned_char_length))]
    # create lemmatized cleaned column
    df['lemmatized_readme'] = df.cleaned_readme.apply(p.lemmatize)
    # order and rename columns for preference
    cols = ['repo', 'readme_contents', 'cleaned_readme',
            'lemmatized_readme', 'original_char_length',
            'cleaned_char_length', 'pct_char_removed',
            'natural_language', 'language']
    df = df[cols].reset_index(drop=True)
    df = df.rename(columns={'repo':'repository',
                            'readme_contents':'original_readme',
                            'language':'programming_language'})

    return df


def encode_target(df):
    '''
    '''

    # assign numerical values for each programming language
    df['target_class'] = np.where(df.programming_language == \
                                        'Python', 0, -1)
    df['target_class'] = np.where(df.programming_language == \
                                        'JavaScript', 1, df.target_class)
    df['target_class'] = np.where(df.programming_language == \
                                        'Jupyter Notebook', 2, df.target_class)
    df['target_class'] = np.where(df.programming_language == \
                                        'HTML', 3, df.target_class)
    df['target_class'] = np.where(df.programming_language == \
                                        'R', 4, df.target_class)
    df['target_class'] = np.where(df.programming_language == \
                                        'TypeScript', 5, df.target_class)
    # set data type as int
    df.target_class = df.target_class.astype(int)

    return df


def split_data(df):
    '''
    '''

    # split test out from DataFrame
    train_validate, test = train_test_split(df, test_size=0.2,
                                            random_state=19,
                                            stratify=df.target_class)
    # split train and validate data sets
    train, validate = train_test_split(train_validate, test_size=0.25,
                                       random_state=19,
                                       stratify=train_validate.target_class)
    # split each into X, y sets
    X_train = train.drop(columns=['programming_language', 'target_class'])
    y_train = train[['programming_language', 'target_class']]
    X_validate = validate.drop(columns=['programming_language', 'target_class'])
    y_validate = validate[['programming_language', 'target_class']]
    X_test = test.drop(columns=['programming_language', 'target_class'])
    y_test = test[['programming_language', 'target_class']]

    return (X_train, y_train,
            X_validate, y_validate,
            X_test, y_test)


def wrangle_github_repos(new_pickles=False, get_new_links=False,
                                             number_of_pages=25):
    '''
    '''

    if get_new_links == True or isfile('data2.json') == False:
        get_new_links = True
        get_repo_links(number_of_pages=number_of_pages)
        data = scrape_github_data()
        json.dump(data, open('data2.json', 'w'), indent=1)
    # if file does not exist, or is overwritten, read in and pickle
    if (isfile('repos.pickle') == False or 
                    get_new_links == True or
                    new_pickles == True):
        df = prep_github_repos()
        df = polish_github_repos(df)
        df = encode_target(df)
        make_pickles(df, 'repos')
        X_train, y_train, \
        X_validate, y_validate, \
        X_test, y_test = split_data(df)
    # if file exists, unpickle
    df = open_pickles('repos')
    X_train, y_train, \
    X_validate, y_validate, \
    X_test, y_test = split_data(df)

    return (X_train, y_train,
            X_validate, y_validate,
            X_test, y_test)
