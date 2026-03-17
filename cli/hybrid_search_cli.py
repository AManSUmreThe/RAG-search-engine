import argparse

from lib.hybrid_search import (
    rrf_search,
    weighted_search,
    normalize
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser('normalize', help='Pass list of scores to mormalize')
    normalize_parser.add_argument('score_list', type=float, nargs='+', help='Input list of scores')

    weighted_search_parser = subparsers.add_parser('weighted_search', help="hybrid weighted search")
    weighted_search_parser.add_argument('query', type=str, help="Search query")
    weighted_search_parser.add_argument('--alpha', type=float, default=0.5 ,help='percentage keyword/semantic results in search')
    weighted_search_parser.add_argument('--limit', type=int, default=10, help='Maximun number of results')

    rrf_search_parser = subparsers.add_parser('rrf_search', help="hybrid recipocal rank fusion search")
    rrf_search_parser.add_argument('query', type=str, help="Search query")
    rrf_search_parser.add_argument('--k', type=int, default=60 ,help='K hyperparameter of rrf score')
    rrf_search_parser.add_argument('--limit', type=int, default=10, help='Maximun number of results')


    args = parser.parse_args()

    match args.command:
        case 'rrf_search':
            results = rrf_search(args.query,args.k,args.limit)
            for idx,res in enumerate(results,start=1):
                print(f"{idx}. {res['title']}")
                print(f"RRF Score: {res['rrf_score']:.3f}")
                print(f"BM25 Rank: {res['bm25_rank']}, Semantic Rank: {res['semantic_rank']}")
                print(res['document'])
        case 'weighted_search':
            results = weighted_search(args.query,args.alpha,args.limit)
            for idx,res in enumerate(results,start=1):
                print(f"{idx}. {res['title']}")
                print(f"Hybrid Score: {res['hybrid_score']:.3f}")
                print(f"BM25: {res['bm25_score']:.3f}, Semantic: {res['semantic_score']:.3f}")
                print(res['document'])
        case 'normalize':
            scores = normalize(list(args.score_list))
            for score in scores:
                print(f"* {score:.4f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()