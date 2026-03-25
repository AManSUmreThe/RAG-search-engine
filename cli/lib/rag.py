from lib.llm import generate_response
from lib.search_utils import PROMPTS_PATH


def format_results(results):
    temp = "<movie>\n<title>{title}</title>\n<desicription>{description}</description>\n</movie>"
    formatted_results = ""

    for res in results:
        formatted_results += temp.format(
            title = res['title'],
            description = res['document']
        )
    
    return formatted_results

# Query Answering methods
def answer_query(query,results):
    with open(PROMPTS_PATH/'rag.md', 'r') as f:
        file = f.read()

    formatted_results = format_results(results)
    prompt = file.format(
        query = query,
        docs = formatted_results
    )

    return generate_response(prompt)

def summarize(query,results):
    with open(PROMPTS_PATH/'summarize.md') as f:
        file = f.read()
    
    formatted_results = format_results(results)

    prompt = file.format(
        query = query,
        results = formatted_results
    )

    return generate_response(prompt)