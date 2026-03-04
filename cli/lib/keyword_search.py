from lib.search_utils import load_movies_data, load_stopwords, CACHE_PATH

import string
import os
import pickle

from collections import defaultdict
from nltk.stem import PorterStemmer
# initializing stemmer
stemmer = PorterStemmer()

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.index_path = CACHE_PATH/'index.pkl'
        self.docmap_path = CACHE_PATH/'docmap.pkl'

    def __add_document(self, doc_id, text):
        tokens = tokenize(text)
        for token in set(tokens):
            self.index[token].add(doc_id)

    def get_documents(self, term):
        return sorted(list(self.index[term]))
    
    def build(self):
        movies = load_movies_data()
        for movie in movies:
            doc_id = movie['id']
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id,text)
            self.docmap[doc_id] = movie
    
    def save(self):
        # making dir if dosen't exists
        os.makedirs(CACHE_PATH, exist_ok=True)

        # saving indexs in index.pkl
        with open(self.index_path,'wb') as f:
            pickle.dump(self.index,f)
        
        # saving movies in docmap.pkl
        with open(self.docmap_path,'wb') as f:
            pickle.dump(self.docmap,f)

    def load(self):

        # loading indexs from index.pkl
        with open(self.index_path,'r') as f:
            self.index = pickle.load(f)      
        # loading movies from docmap.pkl
        with open(self.docmap_path,'r') as f:
            self.docmap = pickle.load(f)
        # return index , docmap

# puncuation
def puncuate(text):
    text = text.lower()
    text = text.translate(str.maketrans("","",string.punctuation))
    return text

# tokenization
def tokenize(text):
    # loading stopwords list 
    stopwords = load_stopwords()
    text = puncuate(text)
    tokens = []
    for token in text.split():
        # removing stopwords
        if token and (token not in stopwords):
            token = stemmer.stem(token)
            tokens.append(token)
    return tokens

# Checking match between keyword and movie tokens
def check_match(keywords,movie_tokens):
    for keyword in keywords:
        for movie_token in movie_tokens:
            if keyword in movie_token:
                return True
    return False
    
def search_movies(keywords,n_results = 5):
    movies = load_movies_data()
    results = []
    keywords = tokenize(keywords)
    for movie in movies:
        movie_tokens = tokenize(movie['title'])
        if check_match(keywords,movie_tokens):
            results.append(movie)
        if len(results) == n_results:
            break

    return results

def build_index():
    idx = InvertedIndex()
    idx.build()
    idx.save()

    # docs = idx.get_documents('merida')
    # print(f"First document for token 'merida' = {docs[0]}")
