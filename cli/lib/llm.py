import os
import time
import json
from collections import defaultdict

from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from google import genai

from lib.search_utils import HF_TOKEN, PROMPTS_PATH

def generate_response(content,model_name="gemma-3-27b-it"):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model= model_name,contents= content)
    # print(response.text)
    # print(response.usage_metadata.prompt_token_count)
    # ### known bug returns None 
    # print(response.usage_metadata.candidates_tokens_details)
    # ### ignore if works correctly
    return response.text

# def correct_spelling(query):
    
#     with open(PROMPTS_PATH/'spellings.md','r') as f:
#         prompt = f.read()
#     prompt = prompt.format(query=query)
#     # print(prompt)
#     return generate_response(prompt)
    
# def rewrite_query(query):

#     with open(PROMPTS_PATH/'rewrite.md','r') as f:
#         prompt = f.read()
#     prompt = prompt.format(query=query)
#     # print(prompt)
#     return generate_response(prompt)

# def expand_query(query):
#     with open(PROMPTS_PATH/'expansion.md','r') as f:
#         prompt = f.read()
#     prompt = prompt.format(query=query)
#     # print(prompt)
#     return generate_response(prompt)

def augment_query(query,type):
    with open(PROMPTS_PATH/f'{type}.md','r') as f:
        prompt = f.read()
    prompt = prompt.format(query=query)
    # print(prompt)
    return generate_response(prompt)

def individual_rerank_results(documents,query,type):
    with open(PROMPTS_PATH/'individual.md','r') as f:
        prompt = f.read()
    results = []
    for doc in documents:

        rank = generate_response(prompt.format(
            query = query,
            title = doc['title'],
            document = doc['document']
        ))

        try:
            rank = int(rank)
        except:
            print(f"LLM Falied to Rank the movie {doc['title']} with {rank}")
            rank = 0
        results.append({**doc,'rerank': rank})
        print(rank,doc['title'])
        time.sleep(15)
    return sorted(results,key = lambda x:x['rerank'], reverse=True)

def batch_rerank_results(documents, query):
    doc_string = '''<movie>\n<ID>{id}</ID><title>{title}</title>\n<description>{document}</description>\n</movie>'''
    doc_str_list = ''
    for doc in documents:
        doc_str_list += doc_string.format(
            id = doc['id'],
            title = doc['title'],
            document = doc['document']
        )
    
    with open(PROMPTS_PATH/'batch.md', 'r') as f:
         prompt = f.read()
    prompt=prompt.format(
        query=query,
        doc_list_str = doc_str_list
    )
    response = generate_response(prompt)
    # print(response)
    try:
        response_parse = json.loads(response)
        # print(response_parse)
    except:
        # print(response_parse)
        print('failed to load response in Json.\n LLM returned invalid json format')
        return []
    
    results = []
    docmap = {doc['id']:doc for doc in documents}

    for idx,doc_id in enumerate(response_parse, start=1):
        doc = docmap[doc_id]
        results.append({**doc,'rerank':idx})
    
    return results
    # for idx,doc in enumerate(documents):
    #     try:
    #         rank = int(response_parse[idx])
    #     except:
    #         rank = float('inf')
    #         print(f'failed to load rank of {doc['title']}.\n invalid rank {response_parse[idx]}(not a number)')
    #     results.append({**doc,'rerank':rank})

    # return sorted(results, key= lambda x:x['rerank'])

def cross_encoding_results(documents,query):
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2",token=HF_TOKEN)
    pairs = get_pairs(documents,query)
    scores = cross_encoder.predict(pairs)
    # print(scores)
    results = []
    for idx,doc in enumerate(documents):
        results.append({**doc,'rerank':scores[idx]})
    return sorted(results,key = lambda x:x['rerank'], reverse=True)

def get_pairs(documents,query):
    pairs = []
    for doc in documents:
        pairs.append([query,f'{doc['title']}-{doc['document']}'])
    return pairs