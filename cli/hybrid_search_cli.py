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

    rrf_search_parser.add_argument("--enhance",
                            type=str,
                            choices=["spell","rewrite","expand"],
                            help="Query enhancement method",
                            )
    rrf_search_parser.add_argument("--rerank-method",
                                   type=str,
                                   choices=["individual","batch","cross_encoder"],
                                   help="Re ranking results with llm"
                                   )


    args = parser.parse_args()

    match args.command:
        case 'rrf_search':
            if args.rerank_method:
                print(f'Re ranking top {args.limit} results using {args.rerank_method} method')

            results = rrf_search(args.query,args.k,args.limit,args.enhance,args.rerank_method)
            
            for idx,res in enumerate(results,start=1):
                print(f"{idx}. {res['title']}")
                if res['rerank']: 
                    print(f'Re-rank Rank: {res['rerank']:.4f}')
                else:
                    print(f"RRF Score: {res['rrf_score']:.3f}")
                print(f"BM25 Rank: {res['bm25_rank']}, Semantic Rank: {res['semantic_rank']}")
                print(res['document'][:100])
        case 'weighted_search':
            results = weighted_search(args.query,args.alpha,args.limit)
            for idx,res in enumerate(results,start=1):
                print(f"{idx}. {res['title']}")
                print(f"Hybrid Score: {res['hybrid_score']:.4f}")
                print(f"BM25: {res['bm25_score']:.3f}, Semantic: {res['semantic_score']:.3f}")
                print(res['document'][:100])
        case 'normalize':
            scores = normalize(list(args.score_list))
            for score in scores:
                print(f"* {score:.4f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()