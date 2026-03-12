#!/usr/bin/env python3

from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    chunk_query,
    semantic_chunk_query,
    build_embed_chunks,
    search
    )
import argparse

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify Embedding model")

    verify_embeddings_parser = subparsers.add_parser('verify_embeddings',help='Get Embeddings of given query')
    # verify_embeddings_parser.add_argument("query", type=str, help="Search query")

    embed_chunks_parser = subparsers.add_parser('embed_chunks', help="build chunk embeddings and metadata")


    embed_text_parser = subparsers.add_parser('embed_text',help='Get Embeddings of given text')
    embed_text_parser.add_argument("query", type=str, help="Search query")

    embed_query_parser = subparsers.add_parser('embedquery',help='Get Embeddings of given query')
    embed_query_parser.add_argument("query", type=str, help="Search query")

    chunk_parser = subparsers.add_parser('chunk',help='Get Embeddings of given query')
    chunk_parser.add_argument("query", type=str, help="Search query")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Define Chunk size")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="chunk overlaping size")

    semantic_chunk_parser = subparsers.add_parser('semantic_chunk',help='Get Embeddings of given query')
    semantic_chunk_parser.add_argument("query", type=str, help="Search query")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Define Chunk size")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="chunk overlaping size")

    search_parser = subparsers.add_parser('search',help='Search the documents for query')
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int,default=5, help="Search result limit")



    args = parser.parse_args()

    match args.command:
        case "search":
            search(args.query,args.limit)

        case "embed_chunks":
            build_embed_chunks()

        case "chunk":
            chunk_query(args.query, args.chunk_size, args.overlap)
        
        case "semantic_chunk":
            semantic_chunk_query(args.query, args.max_chunk_size, args.overlap)
        case "verify":
            model = verify_model()
            print(f"Model loaded: {model}") 
            print(f"Max sequence length: {model.max_seq_length}")

        case "verify_embeddings":
            verify_embeddings()

        case 'embed_text':
            embedding = embed_text(args.query)
            print(f"Text: {args.query}")
            print(f"First 3 dimensions: {embedding[:3]}")
            print(f"Dimensions: {embedding.shape[0]}")

        case 'embedquery':
            embed_query_text(args.query)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()