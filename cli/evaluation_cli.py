import argparse

import json
import time
from lib.search_utils import ROOT, PROMPTS_PATH
from lib.llm import generate_response
from lib.hybrid_search import rrf_search

def llm_evaluation(query,results):
    # results = results[:limit]
    formatted_results = format_results(results)
    
    with open(PROMPTS_PATH/'llm_eval.md', 'r') as f:
        file = f.read()
    prompt = file.format(
        query = query,
        formatted_results = formatted_results
    )
    response = generate_response(prompt)
    time.sleep(15)

    try:
        response_parse = json.loads(response)
        # print(response_parse)
    except:
        # print(response_parse)
        print('failed to load response in Json.\n LLM returned invalid json format')
        return []
    
    return response_parse
    

def format_results(results):
    titles = []

    for res in results:
        titles.append(res['title'])

    formatted_results = ", ".join(titles)
    return formatted_results

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )
    parser.add_argument("--llm-eval",
                        action='store_true'
                        )

    args = parser.parse_args()
    limit = args.limit
    # run evaluation logic here
    with open(ROOT/'data'/'golden_dataset.json','r') as f:
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
        print(f"Query: {query}, limit:{limit}")

        if args.llm_eval:
            llm_score = llm_evaluation(query,results)
            print(llm_score)
            print('LLM evaluation for results:')
            for idx,res in enumerate(results):
                print(f'{idx+1}. {res['title']}: {llm_score[idx]}/3')
                if res['title'] in relevant_docs:
                    relevant_retrieved += 1
        else:
            for res in results:
                if res['title'] in relevant_docs:
                    relevant_retrieved += 1

        precision = relevant_retrieved / total_retrieved
        recall = relevant_retrieved / len(relevant_docs)
        f1 = 2 * (precision * recall) / (precision + recall)
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