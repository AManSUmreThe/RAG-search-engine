import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

from google import genai

client = genai.Client(api_key=api_key)

content = "Why is gemma 3 27b returning None .usage_metadata.candidates_token_count"
model_name = "gemma-3-27b-it"
response = client.models.generate_content(
    model= model_name,
    contents= content,
)

print(response.text)
print(response.usage_metadata.prompt_token_count)

# known bug returns None 
print(response.usage_metadata.candidates_tokens_details)
# ignore if works correctly