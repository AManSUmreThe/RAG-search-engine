from lib.search_utils import load_movies_data
import string

# puncuation
def puncuate(text):
    text = text.lower()
    text = text.translate(str.maketrans("","",string.punctuation))
    return text

# tokenization
def tokenize(text):
    text = puncuate(text)
    tokens = [token for token in text.split() if token]
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