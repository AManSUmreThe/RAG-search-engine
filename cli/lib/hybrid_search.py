from collections import defaultdict
import os

from lib.keyword_search import InvertedIndex
from lib.semantic_search import ChunkedSemanticSearch
from lib.search_utils import load_movies_data


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        # raise NotImplementedError("Weighted hybrid search is not implemented yet.")
        bm25_results = self._bm25_search(query,limit*500)
        semantic_results = self.semantic_search.search_chunks(query,limit*500)

        results = self.get_weighted_results(bm25_results,semantic_results,alpha)
        # print(results[:limit])
        return results[:limit]

    def hybrid_score(self,bm25_score, semantic_score, alpha=0.5):
        return alpha * bm25_score + (1 - alpha) * semantic_score
    
    def normalize_results(self,results):
        scores = [res['score'] for res in results]
        norm_scores = normalize(scores)

        for idx,res in enumerate(results):
            res['norm_score'] = norm_scores[idx]

        return results
    def get_weighted_results(self,bm25_results,semantic_results,alpha=0.5):
        # print(bm25_results[:5])
        bm25_norm_res = self.normalize_results(bm25_results)
        semantic_norm_res = self.normalize_results(semantic_results)
        
        combined_res = defaultdict()
        for norm in bm25_norm_res:
            combined_res[norm["id"]] = {
                'id': norm['id'],
                'title': norm['title'],
                'document': norm['document'],
                'bm25_score': norm['norm_score'],
                'semantic_score': 0
            }
        
        for norm in semantic_norm_res:
            doc = combined_res[norm["id"]]
            if doc:
                doc['semantic_score'] = norm['norm_score']
            else:
                doc = {
                'id': norm['id'],
                'title': norm['title'],
                'document': norm['document'],
                'bm25_score': 0,
                'semantic_score': norm['norm_score']
                }

        for res in combined_res.values():
            res['hybrid_score'] = self.hybrid_score(res['bm25_score'],res['semantic_score'],alpha)

        # # print(combined_res[:10])
        # for idx,res in enumerate(combined_res):
        #     if idx == 5:
        #         break
        #     print(combined_res[res])
        
        return sorted(combined_res.values(), key= lambda x: x['hybrid_score'],reverse=True)

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

def weighted_search(query, alpha, limit=5):
    documents = load_movies_data()
    hybrid = HybridSearch(documents)

    results = hybrid.weighted_search(query,alpha,limit)

    return results

def normalize(scores):
    if not scores or len(scores) == 0 :
        return []
    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.]*len(scores)
    
    score_range =  max_score-min_score

    return [(score - min_score)/score_range for score in scores]