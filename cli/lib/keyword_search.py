from lib.search_utils import load_movies_data
import string
def puncuate(text):
    text = text.lower()
    text = text.translate(str.maketrans("","",string.punctuation))
    return text

def search_movies(keyword,n_results = 5):
    movies = load_movies_data()
    results = []
    keyword = puncuate(keyword)
    for movie in movies:
        if keyword in puncuate(movie['title']):
            results.append(movie)
        if len(results) == n_results:
            break

    return results