import argparse
from lib.hybrid_search import rrf_search, weighted_search

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            # do RAG stuff here
            results = rrf_search(query,k=60,limit=3,enhances=None,rerank=None)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()