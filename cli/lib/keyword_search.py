import math

from lib.search_utils import(
    load_movies_data,
    load_stopwords,
    CACHE_PATH,
    BM25_B,
    BM25_K1
)

import string
import os
import pickle

from collections import defaultdict, Counter
from nltk.stem import PorterStemmer
# initializing stemmer
stemmer = PorterStemmer()

class InvertedIndex:
    def __init__(self):
        # class attributes
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths = defaultdict()
        # save file paths
        self.index_path = CACHE_PATH/'index.pkl'
        self.docmap_path = CACHE_PATH/'docmap.pkl'
        self.term_frequencies_path = CACHE_PATH/'term_frequencies.pkl'
        self.doc_lengths_path = CACHE_PATH/"doc_lengths.pkl"
    
    def __get_avg_doc_length(self) -> float:
        # sum of all doc lengths
        total_sum = sum(list(self.doc_lengths.values()))
        # total no. of documents
        total_docs = len(self.doc_lengths)
        if total_docs == 0:
            return 0

        return total_sum/total_docs
    def bm25_search(self, query, limit=5):
        tokens = tokenize(query)
        # BM25 scores
        scores = {}

        # adding scores to doc id based on query token
        for doc_id in self.docmap:
            score = 0
            for token in tokens:
                score += self.bm25(doc_id,token)
            scores[doc_id] = score
        
        # sorting the docs based on their bm25 scores 
        sorted_docs = sorted(scores.items(),key= lambda x:x[1], reverse=True)

        results = []

        for doc_id,score in sorted_docs[:limit]:
            doc= self.docmap[doc_id]
            results.append(
                {
                    "id": doc_id,
                    "title": doc['title'],
                    'document':doc['description'][:100],
                    "score": score
                }
            )
        
        return results


    def get_tf(self,doc_id,term):
        # tf_tokens = self.term_frequencies[doc_id]
        term = tokenize(term)
        if len(term) != 1:
            raise ValueError("More than 1 token found while searching term frequncies")
        return self.term_frequencies[doc_id][term[0]]
    
    def get_idf(self,term):
        term = tokenize(term)
        if len(term) != 1:
            raise ValueError("More than 1 token found while searching term frequncies")
        term = term[0]
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index[term])

        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))
    
    def get_tf_idf(self,doc_id,term):
        tf = self.get_tf(doc_id,term)
        idf = self.get_idf(term)

        return tf*idf
    
    def get_bm25_idf(self, term: str) -> float:
        term = tokenize(term)
        if len(term) != 1:
            raise ValueError("More than 1 token found while searching term frequncies")
        term = term[0]
        N = len(self.docmap)
        df = len(self.index[term])

        return math.log((N - df + 0.5) / (df + 0.5) + 1)
    
    def get_bm25_tf(self,doc_id,term: str,k1=BM25_K1,b=BM25_B):
        tf = self.get_tf(doc_id,term)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()

        # Length normalization factor
        length_norm = 1 - b + b * (doc_length / avg_doc_length)

        # Apply to term frequency
        tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm)

        return tf_component
    
    def __add_document(self, doc_id, text):
        tokens = tokenize(text)
        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)
        for token in set(tokens):
            self.index[token].add(doc_id)
        

    def get_documents(self, term):
        return sorted(list(self.index[term]))
    
    def bm25(self,doc_id,term: str,k1=BM25_K1,b=BM25_B):

        bm25_tf = self.get_bm25_tf(doc_id,term,k1,b)
        bm25_idf = self.get_bm25_idf(term)

        return bm25_tf*bm25_idf
    
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
        # saving document lengths
        with open(self.doc_lengths_path,'wb') as f:
            pickle.dump(self.doc_lengths,f)

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
        #loading document lengths
        with open(self.doc_lengths_path,'rb') as f:
            self.doc_lengths = pickle.load(f)

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

# # Checking match between keyword and movie tokens
# def check_match(keywords,movie_tokens):
#     for keyword in keywords:
#         for movie_token in movie_tokens:
#             if keyword in movie_token:
#                 return True
#     return False
def build_index():
    idx = InvertedIndex()
    idx.build()
    idx.save()

    # docs = idx.get_documents('merida')
    # print(f"First document for token 'merida' = {docs[0]}")

def bm25_search(query,limit=5):
    idx = InvertedIndex()
    idx.load()
    # tokens = tokenize(query)
    # # BM25 scores
    # scores = {}
    # # adding scores to doc id based on query token
    # for doc_id in idx.docmap:
    #     score = 0
    #     for token in tokens:
    #         score += idx.bm25(doc_id,token)
    #         scores[doc_id] = score    
    # # sorting the docs based on their bm25 scores 
    # sorted_docs = sorted(scores.items(),key= lambda x:x[1], reverse=True)
    # results = []
    # for doc_id,score in sorted_docs[:limit]:
    #     title = idx.docmap[doc_id]['title']
    #     results.append(
    #         {
    #             "doc_id": doc_id,
    #             "title": title,
    #             "score": score
    #         }
    #     )
    results = idx.bm25_search(query,limit)
    return results

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

def search_tf(doc_id,token):
    idx = InvertedIndex()
    idx.load()
    return idx.term_frequencies[doc_id][token]

def search_idf(term):
    idx = InvertedIndex()
    idx.load()

    return idx.get_idf(term)

def search_tf_idf(doc_id,term):
    idx = InvertedIndex()
    idx.load()

    return idx.get_tf_idf(doc_id,term)

def search_BM25_idf(term):
    idx = InvertedIndex()
    idx.load()

    return idx.get_bm25_idf(term)

def search_BM25_tf(doc_id,term,k1,b):
    idx = InvertedIndex()
    idx.load()

    return idx.get_bm25_tf(doc_id,term,k1,b)