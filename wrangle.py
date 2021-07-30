#Z0096

# import standard libraries
import pandas as pd
import re

# import file managers
from os.path import isfile
import pickle

# import json handler
import json

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


def remove_code_snippest(df):
    '''
    '''

    # remove HTML and markdown code
    df['cleaned'] = df.readme_contents.apply(lambda row: re.sub(r'(\<.+\>)', '', row))
    df['cleaned'] = df.cleaned.apply(lambda row: re.sub(r'!?(\[.+\])(\(.+\))?', '', row))
    # remove coded spaces and page breaks
    df['cleaned'] = df.cleaned.apply(lambda row: re.sub(r'(\n+)', ' ', row))
    df['cleaned'] = df.cleaned.apply(lambda row: re.sub(r'(&nbsp;+)', '', row))
    # remove naked hyperlinks
    df['cleaned'] = df.cleaned.apply(lambda row: re.sub(r'http\S+', '', row))
    # replaced hyphens with spaces
    df['cleaned'] = df.cleaned.apply(lambda row: re.sub(r'-', ' ', row))
    
    return df


def extensive_clean(df):
    '''
    '''
    
    # run cleaner functions on text data
    df['cleaned'] = df.cleaned.apply(lambda row: p.remove_stopwords(
                                                 p.tokenize(
                                                 p.basic_clean(row))))
    
    return df


def create_char_counts(df):
    '''
    '''
    
    # create character counts for original and cleaned text
    df['original_char_length'] = df.readme_contents.apply(lambda row: len(row))
    df['cleaned_char_length'] = df.cleaned.apply(lambda row: len(row))
    
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


def wrangle_github_repos(new_pickles=False, get_new_links=False,
                                             number_of_pages=25):
    '''
    '''

    if get_new_links == True or isfile('data2.json') == False:
        get_repo_links(number_of_pages=number_of_pages)
        data = scrape_github_data()
        json.dump(data, open('data2.json', 'w'), indent=1)
    # if file does not exist, or is overwritten, read in and pickle
    if (isfile('repos.pickle') == False or
                  get_new_links == True or
                      new_pickles == True):
        # load data into DataFrame
        df = open_json_data()
        # filter data to only English results
        df = get_english_only(df)
        # remove HTML, markdown, etc.
        df = remove_code_snippest(df)
        # remove non-ascii chars and stopwords, tokenize
        df = extensive_clean(df)
        # create character count columns
        df = create_char_counts(df)
        # create percent change column
        df = create_pct_changed(df)
        # filter for programming languages
        df = filter_language(df)
        # remove cleaned char count outliers
        df = df[df.cleaned_char_length.isin(
                    p.filter_iqr_outliers(df.cleaned_char_length))]
        # create lemmatized cleaned column
        df['lemmatized'] = df.cleaned.apply(p.lemmatize)
        # order columns for preference
        cols = ['repo', 'readme_contents', 'cleaned',
                'lemmatized', 'original_char_length',
                'cleaned_char_length', 'pct_char_removed',
                'natural_language', 'language']
        df = df[cols]
        make_pickles(df, 'repos')
        return df
    # if file exists, unpickle
    df = open_pickles('repos')

    return df