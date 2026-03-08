from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def generate_embedding(self, text: str):
        if not text or not text.strip():
            raise ValueError("cannot embed Empty text")
        return self.model.encode([text])[0]

def verify_model():
    semantic = SemanticSearch()
    
    return semantic.model

def embed_text(text):
    semantic = SemanticSearch()
    return semantic.generate_embedding(text)
