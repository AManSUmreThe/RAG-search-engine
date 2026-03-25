import argparse

import json
import time
from lib.search_utils import EVAL_PATH
from lib.hybrid_search import rrf_search

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit
    # run evaluation logic here
    with open(EVAL_PATH,'r') as f:
        file = f.read()
    test_data = json.loads(file)
    test_data = test_data['test_cases']

    for test in test_data:
        query = test['query']
        relevant_docs = test["relevant_docs"]
        results = rrf_search(query,60,limit,rerank='cross_encoder')
        results = results[:limit]
        top_K = len(results)

        total_retrieved = len(results)
        relevant_retrieved = 0
        for res in results:
            if res['title'] in relevant_docs:
                relevant_retrieved += 1

        precision = relevant_retrieved / total_retrieved
        recall = relevant_retrieved / len(relevant_docs)
        f1 = 2 * (precision * recall) / (precision + recall)
        print(f"Query: {query}, limit:{limit}")
        print(f'''
Total retrieved docs: {total_retrieved}
Relevant docs: {relevant_retrieved}
Relevant docs in golden dataset: {len(relevant_docs)} 

Result Precision@{top_K}: {precision:.4f}
Recall@{top_K}: {recall:.4f}
F1-score@{top_K}: {f1:.4f}
''')
        print("_"*60)

        # time.sleep(30)

if __name__ == "__main__":
    main()