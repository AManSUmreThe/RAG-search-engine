import os
import time
from collections import defaultdict

from dotenv import load_dotenv

from google import genai

from lib.search_utils import PROMPTS_PATH

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

def rerank_results(documents,query,type):
    with open(PROMPTS_PATH/f'{type}.md','r') as f:
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
        