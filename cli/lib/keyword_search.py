from lib.search_utils import load_movies_data

def search_movies(keyword,n_results = 5):
    movies = load_movies_data()
    results = []
    for movie in movies:
        if keyword in movie['title']:
            results.append(movie)
        if len(results) == n_results:
            break

    return results