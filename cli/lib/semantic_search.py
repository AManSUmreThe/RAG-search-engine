import numpy as np
import re
import json
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

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self):
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None

        self.chunk_embeddings_path = CACHE_PATH/"chunk_embeddings.npy"
        self.chunk_metadata_path = CACHE_PATH/"chunk_metadata.json"

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        self.docmap = {doc['id']:doc for doc in documents}

        all_chunks = []
        chunks_metadata = []
        for doc in documents:
            if doc['description'] == '':
                continue
            chunks = semantic_chunk_query(doc['description'], max_chunk_size=4 , overlap= 1)
            all_chunks += chunks
            for idx in range(len(chunks)):
                chunks_metadata.append({
                    'movie_idx':doc['id'],
                    'chunk_id': idx,
                    'total_chunks': len(chunks)
                })
        self.chunk_embeddings = self.model.encode(all_chunks)
        self.chunk_metadata = chunks_metadata
        
        np.save(self.chunk_embeddings_path,self.chunk_embeddings)
        with open(self.chunk_metadata_path,'w') as f:
            json.dump({"chunks": chunks_metadata, "total_chunks": len(all_chunks)}, f, indent=2)
        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents):
        self.documents = documents
        self.docmap = {doc['id']:doc for doc in documents}

        if self.chunk_embeddings_path.exists() and self.chunk_metadata_path.exists():
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path, 'r') as f:
                self.chunk_metadata =json.load(f)
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)

def verify_model():
    semantic = SemanticSearch()
    
    return semantic.model
def verify_embeddings():
    documents = load_movies_data()
    semamtic = SemanticSearch()
    embeddings = semamtic.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def build_embed_chunks():
    chunkedSemantic = ChunkedSemanticSearch()
    documents = load_movies_data()
    embeddings = chunkedSemantic.load_or_create_chunk_embeddings(documents)

    print(f"Generated {len(embeddings)} chunked embeddings")
    

def embed_text(text):
    semantic = SemanticSearch()
    return semantic.generate_embedding(text)

def embed_query_text(query):
    semantic = SemanticSearch()
    embedding = semantic.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def chunk_query(query,chunk_size,overlap):
    words = query.split()
    chunks = []

    step_size = chunk_size - overlap
    # print(step_size)
    for i in range(0,len(words),step_size):
        chunk_words = words[i:i+chunk_size]
        if len(chunk_words) <= overlap:
            break
        chunks.append(" ".join(chunk_words))
        # print(chunks)
    print(f"Chunking {len(query)} characters")
    for idx,chunk in enumerate(chunks,start=1):
        print(f"{idx}. {chunk}")

def semantic_chunk_query(query,max_chunk_size,overlap):
    sentences = re.split(r"(?<=[.!?])\s+",query)
    chunks = []
    step_size = max_chunk_size - overlap

    for i in range(0,len(sentences),step_size):
        chunk = sentences[i:i+max_chunk_size]
        if len(chunk) <= overlap:
            break
        chunks.append(" ".join(chunk))
    print(f"Semantic chunking {len(query)} characters")
    for idx,chunk in enumerate(chunks, start=1):
        print(f"{idx}. {chunk}")
    
    return chunks

def search(query,limit=5):
    semamtic = SemanticSearch()
    document = load_movies_data()
    semamtic.load_or_create_embeddings(document)
    search_res = semamtic.search(query,limit)

    for idx,res in enumerate(search_res,start=1):
        print(f"{idx}. {res['title']} (score: {res['score']:.4f})")
        print(res['description'][:100])