"""
A module for obtaining repo readme and language data from the github API.
Before using this module, read through it, and follow the instructions marked
TODO.
After doing so, run it like this:
    python acquire.py
To create the `data.json` file that contains the data.
"""

import os
import json
from typing import Dict, List, Optional, Union, cast
import requests

from env import github_token, github_username

# TODO: Make a github personal access token.
#     1. Go here and generate a personal access token https://github.com/settings/tokens
#        You do _not_ need select any scopes, i.e. leave all the checkboxes unchecked
#     2. Save it in your env.py file under the variable `github_token`
# TODO: Add your github username to your env.py file under the variable `github_username`
# TODO: Add more repositories to the `REPOS` list below.

REPOS = ['nytimes/covid-19-data',
'covid19india/covid19india-react',
'owid/covid-19-data',
'WorldHealthOrganization/app',
'geohot/corona',
'ahmadawais/corona-cli',
'corona-warn-app/cwa-server',
'ExpDev07/coronavirus-tracker-api',
'mhdhejazi/CoronaTracker',
'soroushchehresa/awesome-coronavirus',
'github/covid19-dashboard',
'pomber/covid19',
'FoldingAtHome/coronavirus',
'datasets/covid-19',
'ryansmcgee/seirsplus',
'chrismattmann/tika-python',
'lindawangg/COVID-Net',
'chandrikadeb7/Face-Mask-Detection',
'ImperialCollegeLondon/covid19model',
'cyberman54/ESP32-Paxcounter',
'UCSD-AI4H/COVID-CT',
'makers-for-life/makair',
'neuml/paperai',
'globalcitizen/2019-wuhan-coronavirus-data',
'pallupz/covid-vaccine-booking',
'javieraviles/covidAPI',
'ryansmcgee/seirsplus',
'ayushi7rawat/CoWin-Vaccine-Notifier',
'aatishb/maskmath',
'siaw23/kovid',
'ryo-ma/covid19-japan-web-api',
'Lewuathe/COVID19-SIR',
'saimj7/People-Counting-in-Real-Time',
'lachlanjc/covid19',
'google/exposure-notifications-android', 
'MohGovIL/hamagen-react-native', 
'phildini/stayinghomeclub', 
'rbignon/doctoshotgun',
'OxCGRT/covid-policy-tracker',
'covidpass-org/covidpass',
'AaronWard/covidify',
'MoH-Malaysia/covid19-public',
'GuangchuangYu/nCov2019',
'wcota/covid19br',
'covidatlas/coronadatascraper',
'reichlab/covid19-forecast-hub',
'swsoyee/2019-ncov-japan',
'CITF-Malaysia/citf-public',
'Ank-Cha/Social-Distancing-Analyser-COVID-19',
'VinAIResearch/BERTweet',
'amodm/api-covid19-in',
'JohnCoene/coronavirus',
'GoogleCloudPlatform/covid-19-open-data',
'deepset-ai/COVID-QA',
'bhattbhavesh91/cowin-vaccination-slot-availability',
'ccodwg/Covid19Canada',
'HzFu/COVID19_imaging_AI_paper_list',
'aatishb/covidtrends',
'github/covid-19-repo-data',
'anshumanpattnaik/covid19-full-stack-application',
'open-covid-19/data',
'paulvangentcom/python_corona_simulation',
'fluttercandies/ncov_2019',
'trekhleb/covid-19',
'COVID-19-electronic-health-system/Corona-tracker',
'wobsoriano/covid3d',
'dsfsi/covid19za',
'ayushi7rawat/CoWin-Vaccine-Notifier',
'cyberboysumanjay/APIs',
'nasa-jpl/COVID-19-respirators',
'porames/the-researcher-covid-bot',
'helpwithcovid/covid-volunteers',
'Kamaropoulos/COVID19Py',
'OpenGene/fastv',
'covid-19-net/covid-19-community',
'tarunk04/COVID-19-CaseStudy-and-Predictions',
'vitorbaptista/google-covid19-mobility-reports',
'Rank23/COVID19',
'starschema/COVID-19-data',
'ChrisMichaelPerezSantiago/covid19',
'jeremykohn/rid-covid',
'agallio/ina-covid-bed',
'livgust/covid-vaccine-scrapers',
'DmitrySerg/COVID-19',
'MaksimEkin/COVID19-Literature-Clustering',
'rpandey1234/Covid19Tracker',
'simonw/covid-19-datasette',
'emmadoughty/Daily_COVID-19',
'mr7495/COVID-CTset',
'Omaroid/Covid-19-API',
'OpenCOVID19CoughCheck/CoughCheckApp',
'wobsoriano/2019-ncov-api',
'abd-shoumik/Social-distance-detection',
'Coders-Of-XDA-OT/covid19-status-android',
'CDCgov/covid19healthbot',
'PyTorchLightning/lightning-Covid19',
'boogheta/coronavirus-countries',
'neuml/cord19q',
'cre8ivepark/COVID19DataVisualizationHoloLens2',
'RespiraWorks/Ventilator']
 

headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}

if headers["Authorization"] == "token " or headers["User-Agent"] == "":
    raise Exception(
        "You need to follow the instructions marked TODO in this script before trying to use it"
    )


def github_api_request(url: str) -> Union[List, Dict]:
    response = requests.get(url, headers=headers)
    response_data = response.json()
    if response.status_code != 200:
        raise Exception(
            f"Error response from github api! status code: {response.status_code}, "
            f"response: {json.dumps(response_data)}"
        )
    return response_data


def get_repo_language(repo: str) -> str:
    url = f"https://api.github.com/repos/{repo}"
    repo_info = github_api_request(url)
    if type(repo_info) is dict:
        repo_info = cast(Dict, repo_info)
        return repo_info.get("language", None)
    raise Exception(
        f"Expecting a dictionary response from {url}, instead got {json.dumps(repo_info)}"
    )


def get_repo_contents(repo: str) -> List[Dict[str, str]]:
    url = f"https://api.github.com/repos/{repo}/contents/"
    contents = github_api_request(url)
    if type(contents) is list:
        contents = cast(List, contents)
        return contents
    raise Exception(
        f"Expecting a list response from {url}, instead got {json.dumps(contents)}"
    )


def get_readme_download_url(files: List[Dict[str, str]]) -> str:
    """
    Takes in a response from the github api that lists the files in a repo and
    returns the url that can be used to download the repo's README file.
    """
    for file in files:
        if file["name"].lower().startswith("readme"):
            return file["download_url"]
    return ""


def process_repo(repo: str) -> Dict[str, str]:
    """
    Takes a repo name like "gocodeup/codeup-setup-script" and returns a
    dictionary with the language of the repo and the readme contents.
    """
    contents = get_repo_contents(repo)
    readme_contents = requests.get(get_readme_download_url(contents)).text
    return {
        "repo": repo,
        "language": get_repo_language(repo),
        "readme_contents": readme_contents,
    }


def scrape_github_data() -> List[Dict[str, str]]:
    """
    Loop through all of the repos and process them. Returns the processed data.
    """
    return [process_repo(repo) for repo in REPOS]


if __name__ == "__main__":
    data = scrape_github_data()
    json.dump(data, open("data2.json", "w"), indent=1)