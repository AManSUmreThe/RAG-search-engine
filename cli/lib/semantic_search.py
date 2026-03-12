import numpy as np
# from collections import defaultdict

from sentence_transformers import SentenceTransformer
from lib.search_utils import (
    CACHE_PATH,
    load_movies_data
)

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.docmap = {}

        self.embeddings_path = CACHE_PATH/"movie_embeddings.npy"
    
    def search(self, query, limit=5):

        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        embedding = self.generate_embedding(query)
        similarities = []

        for doc_emd,doc in zip(self.embeddings,self.documents):
            score = cosine_similarity(embedding,doc_emd)
            similarities.append((score,doc))

        similarities.sort(key=lambda x: x[0], reverse=True)

        results = []
        for (score,doc) in similarities[:limit]:
            results.append({
                'score':score,
                'title':doc['title'],
                'description':doc['description']
            })
        
        return results
    
    def generate_embedding(self, text: str):
        if not text or not text.strip():
            raise ValueError("cannot embed Empty text")
        return self.model.encode([text])[0]
    
    def build_embeddings(self,documents):
        self.documents = documents
        self.docmap = {}
        texts = []
        for doc in documents:
            self.docmap[doc['id']] = doc
            texts.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(texts, show_progress_bar = True)
        np.save(self.embeddings_path,self.embeddings)

        return self.embeddings
    
    def load_or_create_embeddings(self,documents):
        self.documents = documents
        self.docmap = {}
        for doc in documents:
            self.docmap[doc['id']] = doc
        if self.embeddings_path.exists():
            self.embeddings = np.load(self.embeddings_path)
            if len(documents) == len(self.embeddings):
                return self.embeddings
            
        return self.build_embeddings(documents)


def verify_model():
    semantic = SemanticSearch()
    
    return semantic.model
def verify_embeddings():
    documents = load_movies_data()
    semamtic = SemanticSearch()
    embeddings = semamtic.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_text(text):
    semantic = SemanticSearch()
    return semantic.generate_embedding(text)

def embed_query_text(query):
    semantic = SemanticSearch()
    embedding = semantic.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def search(query,limit=5):
    semamtic = SemanticSearch()
    document = load_movies_data()
    semamtic.load_or_create_embeddings(document)
    search_res = semamtic.search(query,limit)

    for idx,res in enumerate(search_res,start=1):
        print(f"{idx}. {res['title']} (score: {res['score']:.2f})")
        print(res['description'][:100])