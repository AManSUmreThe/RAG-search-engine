from collections import defaultdict
import os

from lib.keyword_search import InvertedIndex
from lib.semantic_search import ChunkedSemanticSearch
from lib.search_utils import load_movies_data
from lib.llm import (
    correct_spelling, 
    rewrite_query
    )

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
    
    def rrf_score(self,rank, k):
        return 1 / (k + rank)
    
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
        # raise NotImplementedError("RRF hybrid search is not implemented yet.")
        bm25_results = self._bm25_search(query,limit*500)
        semanti_results = self.semantic_search.search_chunks(query,limit*500)

        results = self.get_rrf_results(bm25_results,semanti_results,k)

        return results[:limit]
    def get_rrf_results(self,bm25_results,semantic_results,k):
        results = defaultdict()
        for rank,res in enumerate(bm25_results,start=1):
            results[res["id"]] = {
                'id': res['id'],
                'title': res['title'],
                'document': res['document'],
                'bm25_rank': rank,
                'semantic_rank': 0
            }
        
        for rank,res in enumerate(semantic_results,start=1):
            doc = results[res["id"]]
            if doc:
                doc['semantic_rank'] = rank
            else:
                doc = {
                'id': res['id'],
                'title': res['title'],
                'document': res['document'],
                'bm25_rank': 0,
                'semantic_rank': rank
                }

        for res in results.values():
            score = .0
            if(res['bm25_rank']>=1):
                score += self.rrf_score(res['bm25_rank'],k)
            if(res['semantic_rank']>=1):
                score += self.rrf_score(res['semantic_rank'],k)
            res['rrf_score'] = score

        sorted_results = sorted(results.values(), key= lambda x: x['rrf_score'],reverse=True)

        return sorted_results

def weighted_search(query, alpha, limit=5):
    documents = load_movies_data()
    hybrid = HybridSearch(documents)

    results = hybrid.weighted_search(query,alpha,limit)

    return results

# rrf search command function
def rrf_search(query, k, enhances, limit=5):

    documents = load_movies_data()
    hybrid = HybridSearch(documents)

    match enhances:
        case 'spell':
            enhanced_query = correct_spelling(query)
            print(f"Enhanced query (spell): '{query}' -> '{enhanced_query}'\n")
            query = enhanced_query
        case 'rewrite':
            rewritten_query = rewrite_query(query)
            print(f"Enhanced query (spell): '{query}' -> '{rewritten_query}'\n")
            query = rewritten_query

    results = hybrid.rrf_search(query,k,limit)

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