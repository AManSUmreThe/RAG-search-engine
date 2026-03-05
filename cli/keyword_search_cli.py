#!/usr/bin/env python3

import argparse
from lib.keyword_search import (
    search_movies,
    build_index,
    search_tf,
    search_idf,
    search_tf_idf,
    search_BM25
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="build IDF indices")

    tf_parser = subparsers.add_parser("tf", help="Get token frequency for given movie id")
    tf_parser.add_argument("doc_id", type=int, help="Movie ID")
    tf_parser.add_argument("term", type=str, help="Search query")

    idf_parser = subparsers.add_parser("idf", help="Get Inverted document frequency for given term")
    idf_parser.add_argument("term", type=str, help="Search query")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF for given term")
    tfidf_parser.add_argument("doc_id", type=int, help="Search query")
    tfidf_parser.add_argument("term", type=str, help="Search query")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    args = parser.parse_args()

    match args.command:
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            
            results = search_movies(args.query)
            for i,movie in enumerate(results):
                print(f"{i}. {movie['title']}")
        case "build":
            build_index()
        case "tf":
            print(search_tf(args.doc_id,args.term))
        case "idf":
            print(f"Inverse document frequency of '{args.term}': {search_idf(args.term):.2f}")
        case "tfidf":
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {search_tf_idf(args.doc_id,args.term):.2f}")
        case "bm25idf":
            print(f"BM25 IDF score of '{args.term}': {search_BM25(args.term):.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()