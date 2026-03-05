#!/usr/bin/env python3

import argparse
from lib.keyword_search import (
    search_movies,
    build_index,
    search_tf
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="build IDF indices")

    tf_parser = subparsers.add_parser("tf", help="Get token frequncy for given movie id")
    tf_parser.add_argument("doc_id", type=int, help="Movie ID")
    tf_parser.add_argument("token", type=str, help="token")


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
            print(search_tf(args.doc_id,args.token))
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()