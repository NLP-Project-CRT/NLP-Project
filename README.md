# This Repository was created to house all applicable files, notebooks, and modules necessary to complete our Natural Language Processing project.

# _GitHub Program Language Predictor Project_



- This project uses a various classification models to predict what programming language a GitHub repository is given the text of the README file.
- In this README we will :
    * Explain what the project is and goals we attempted to reach. 
    * Explain how to reproduce our work. 
    * Contain notes from project planning.

## Goals
- This project aims to predict what programming language a repository is, given the text of the README file. This will be approached as an NLP problem and will try to use various classification models to find the best accuracy score.

## Project Planning
- Trello Board Link:
  - https://trello.com/b/rLuCzaR1/nlp-group-project

**Deliverables:**
1. README.md file containing overall project information, how to reproduce work, and notes from project planning.
2. Jupyter Notebook Report detailing the pipeline process.
3. One or two google slides suitable for a general audience that summarize your findings. Including a well-labelled visualization in slides.

## Key Findings 
* Majority of the Repositories were in the Python programming language.
* Many of the same words were spread across all the top 5 programming languages making for a unreliable prediction.
* Decision Tree was our best performing model over all, however we were expecting better accuracy results on validate datasets.




## Setup this project
* Dependencies
    1. python
    2. pandas
    3. scipy
    4. sklearn
    5. numpy
    6. matplotlib.pyplot
    7. seaborn
    8. wordcloud
    9. requests
* Steps to recreate
    1. Clone this repository
    3. Open `nlp_project_final.ipynb` and run the cells


## Data Context

The data used in this project was obtained through use of GitHub API to obtain contents of README.md from various repositories. This data is obtained through use of the `acquire.py` script and read into the notebooks using the `data2.json` it creates. To reproduce the exact results found within, the JSON file used can be found in the repository contents.

### Data Dictionary
---

Use of this data in a prepared manner throughout this project contains the data defined in the below data dictionary.


| Variable             | Definition                                                  | Data Type |
|----------------------|-------------------------------------------------------------|:---------:|
| repository           | repository location in format `<owner_username>/<repo_name>`  | Object    |
| original_readme      | original contents of README as taken from repository        | Object    |
| cleaned_readme       | original_readme contents processed to only ASCII characters | Object    |
| lemmatized_readme    | cleaned_readme contents lemmatized                          | Object    |
| original_char_length | count of characters in original_readme                      | Integer   |
| cleaned_char_length  | count of characters in cleaned_readme                       | Integer   |
| pct_char_removed     | rounded percent of characters removed from original_readme  | Integer   |
| natural_language     | natural language of README, all values equal `en`           | Object    |
| programming_language | constructed languaged defined as marjoity code by GitHub    | Object    |
| target_class         | integer value corresponding to programming_language         | Integer   |
