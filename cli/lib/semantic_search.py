import numpy as np
# from collections import defaultdict

from sentence_transformers import SentenceTransformer
from lib.search_utils import (
    CACHE_PATH,
    load_movies_data
)

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.docmap = {}

        self.embeddings_path = CACHE_PATH/"movie_embeddings.npy"

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
