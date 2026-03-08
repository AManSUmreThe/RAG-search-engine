#!/usr/bin/env python3

from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings
    )
import argparse

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify Embedding model")

    verify_embeddings_parser = subparsers.add_parser('verify_embeddings',help='Get Embeddings of given query')
    # verify_embeddings_parser.add_argument("query", type=str, help="Search query")


    embed_text_parser = subparsers.add_parser('embed_text',help='Get Embeddings of given query')
    embed_text_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
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
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()