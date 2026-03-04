import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT/'data'/'movies.json'
STOPWORD_PATH = ROOT/'data'/'stopwords.txt'


def load_movies_data() -> list[dict]:
    # loading json
    with open(DATA_PATH, 'r') as file:
        data = json.load(file)
    return data['movies']

def load_stopwords()-> list:
    with open(STOPWORD_PATH,'r') as file:
        stopwords = file.read()
        stopwords = stopwords.splitlines()
        # print(stopwords) 
    return stopwords