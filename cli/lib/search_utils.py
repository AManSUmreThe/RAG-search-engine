import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT/'data'/'movies.json'

def load_movies_data() -> list[dict]:
    # loading json
    with open(DATA_PATH, 'r') as file:
        data = json.load(file)
    return data['movies']