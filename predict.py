#Z0096


####
####
####
####  Run "python predict.py" on command line 
####     and pass URL of raw readme file.
####
####


import re
import pickle
import requests
from prepare import lemmatize, remove_stopwords, remove_stopwords, basic_clean
from wrangle import open_pickles


def remove_code_snippets(readme):
    '''
    Takes argument string and removes obvious HTML and markup code
    present in the readme before returning
    '''

    # remove HTML and markdown code
    readme = re.sub(r'(\<.+\>)', '', readme)
    readme = re.sub(r'!?(\[.+\])(\(.+\))?', '', row)
    # remove coded spaces and page breaks
    readme = re.sub(r'(\n+)', ' ', row)
    readme= re.sub(r'(&nbsp;+)', '', row)
    # remove naked hyperlinks
    readme = re.sub(r'http\S+', '', row)
    # replaced hyphens with spaces
    readme = re.sub(r'-', ' ', row)
    return readme

def predict_readme_lang(readme_url):
    '''
    Takes URL of README document and prints
    the predicted language and the probability.
    '''

    response = requests.get(str(readme_url))
    readme = response.text
    # clean, tokenize, remove stopwords, and lematize
    readme = lemmatize(remove_stopwords(remove_stopwords(basic_clean(readme))))
    # open fitted tfidf object
    tfidf = open_pickles('tfidf')
    readme = tfidf.transform([readme])
    # open fitted model object
    model = open_pickles('model')
    pred = model.predict(readme)
    prob = model.predict_proba(readme)
    # make variable to corresponding langauge to target class
    if pred == 0:
        lang = 'Python'
        prob = prob[0][0]
    elif pred == 1:
        lang = 'JavaScript'
        prob = prob[0][1]
    elif pred == 2:
        lang = 'Jupyter'
        prob = prob[0][2]
    elif pred == 3:
        lang = 'HTML'
        prob = prob[0][3]
    elif pred == 4:
        lang = 'R'
        prob = prob[0][4]
    else:
        lang = 'TypeScript'
        prob = prob[0][5]
    print(f'\n\nThe provided README is predicted as {lang} with {prob:.2%} probability.\n\n')


readme = input('Provide location of readme document: ')
predict_readme_lang(readme)