import argparse
from lib.hybrid_search import rrf_search, weighted_search
from lib.rag import answer_query, summarize

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    summary_parser = subparsers.add_parser(
        "summarize", help="Perform RAG (search + generate answer)"
    )
    summary_parser.add_argument("query", type=str, help="Search query for RAG")
    summary_parser.add_argument("--limit", type=int, default=5, help="Search results limit")


    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            # do RAG stuff here
            results = rrf_search(query,k=60,limit=4,enhances=None,rerank=None)

            print('Search Results: ')
            for idx,res in enumerate(results,start=1):
                print(f'{idx}. {res['title']}')
            
            print('RAG Response')
            response = answer_query(query,results)
            print(response)
        case 'summarize':
            query = args.query
            # do RAG stuff here
            results = rrf_search(query,k=60,limit=args.limit,enhances=None,rerank=None)

            print('Search Results: ')
            for idx,res in enumerate(results,start=1):
                print(f'{idx}. {res['title']}')
            
            print('RAG Response')
            response = summarize(query,results)
            print(response)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()