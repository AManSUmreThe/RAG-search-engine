from lib.search_utils import load_movies_data, load_stopwords
import string

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