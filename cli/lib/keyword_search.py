from lib.search_utils import load_movies_data, load_stopwords, CACHE_PATH

import string
import os
import pickle

from collections import defaultdict, Counter
from nltk.stem import PorterStemmer
# initializing stemmer
stemmer = PorterStemmer()

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)
        self.index_path = CACHE_PATH/'index.pkl'
        self.docmap_path = CACHE_PATH/'docmap.pkl'
        self.term_frequencies_path = CACHE_PATH/'term_frequencies.pkl'

    def __add_document(self, doc_id, text):
        tokens = tokenize(text)
        self.term_frequencies[doc_id].update(tokens)
        for token in set(tokens):
            self.index[token].add(doc_id)
        

    def get_documents(self, term):
        return sorted(list(self.index[term]))
    
    def get_tf(self,doc_id,term):
        # tf_tokens = self.term_frequencies[doc_id]
        term = tokenize(term)
        if len(term) != 1:
            raise ValueError("More than 1 token found while searching term frequncies")
        return self.term_frequencies[doc_id][term[0]]
    
    def build(self):
        movies = load_movies_data()
        for movie in movies:
            doc_id = movie['id']
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id,text)
            self.docmap[doc_id] = movie
            # self.term_frequencies[doc_id] += 1
    
    def save(self):
        # making dir if dosen't exists
        os.makedirs(CACHE_PATH, exist_ok=True)

        # saving indexs in index.pkl
        with open(self.index_path,'wb') as f:
            pickle.dump(self.index,f)
        
        # saving movies in docmap.pkl
        with open(self.docmap_path,'wb') as f:
            pickle.dump(self.docmap,f)

        # saving term frequncies
        with open(self.term_frequencies_path,"wb") as f:
            pickle.dump(self.term_frequencies,f)

    def load(self):

        # loading indexs from index.pkl
        with open(self.index_path,'rb') as f:
            self.index = pickle.load(f)      
        # loading movies from docmap.pkl
        with open(self.docmap_path,'rb') as f:
            self.docmap = pickle.load(f)
        # loading term frequncies
        with open(self.term_frequencies_path,'rb') as f:
            self.term_frequencies = pickle.load(f)

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
    # movies = load_movies_data()
    idx = InvertedIndex()
    idx.load()
    # print(idx.docmap[9])

    seen,results =set(),[]
    keywords = tokenize(keywords)

    # basic searching
    # for movie in movies:
    #     movie_tokens = tokenize(movie['title'])
    #     if check_match(keywords,movie_tokens):
    #         results.append(movie)
    #     if len(results) == n_results:
    #         break

    for keyword in keywords:
        doc_ids = idx.get_documents(keyword)
        for doc_id in doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            results.append(idx.docmap[doc_id])
            if len(results) >= n_results:
                return results
    return results

def build_index():
    idx = InvertedIndex()
    idx.build()
    idx.save()

    # docs = idx.get_documents('merida')
    # print(f"First document for token 'merida' = {docs[0]}")

def search_tf(doc_id,token):
    idx = InvertedIndex()
    idx.load()
    return idx.term_frequencies[doc_id][token]