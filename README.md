# RAG-search-engine

## CLI

This project exposes several `argparse` entrypoints under `cli/` (the entrypoint files are named `*_cli.py`, not `cli.py`).

### `cli/augmented_generation_cli.py`

Subcommand: `rag`

* `python cli/augmented_generation_cli.py rag <query>`
  * `<query>` (str, positional): Search query for RAG

### `cli/evaluation_cli.py`

* `python cli/evaluation_cli.py --limit <int>`
  * `--limit` (int, default: `5`): Number of results to evaluate (k for precision@k, recall@k)
* `python cli/evaluation_cli.py --llm-eval`
  * `--llm-eval` (flag): If set, run LLM-based evaluation over the top results (calls `cli/lib/llm.py::generate_response()` and attempts to parse the response as JSON)

### `cli/hybrid_search_cli.py`

Subcommands: `normalize`, `weighted_search`, `rrf_search`

`normalize`

* `python cli/hybrid_search_cli.py normalize <score_list>...`
  * `<score_list>` (float, 1+ values): Input list of scores

`weighted_search`

* `python cli/hybrid_search_cli.py weighted_search <query> [--alpha <float>] [--limit <int>]`
  * `<query>` (str, positional): Search query
  * `--alpha` (float, default: `0.5`): Keyword/BM25 vs semantic weighting
  * `--limit` (int, default: `10`): Maximum number of results

`rrf_search`

* `python cli/hybrid_search_cli.py rrf_search <query> [--k <int>] [--limit <int>] [--enhance {spell,rewrite,expand}] [--rerank-method {individual,batch,cross_encoder}]`
  * `<query>` (str, positional): Search query
  * `--k` (int, default: `60`): K hyperparameter of the RRF score
  * `--limit` (int, default: `10`): Maximum number of results
  * `--enhance` (str, optional; choices: `spell`, `rewrite`, `expand`): Query enhancement method (loads `prompts/{type}.md` and calls the LLM)
  * `--rerank-method` (str, optional; choices: `individual`, `batch`, `cross_encoder`): Re-ranking method

### `cli/keyword_search_cli.py`

Subcommands: `search`, `bm25search`, `build`, `tf`, `idf`, `tfidf`, `bm25idf`, `bm25tf`

* `python cli/keyword_search_cli.py search <query>`
  * `<query>` (str, positional)
* `python cli/keyword_search_cli.py bm25search <query>`
  * `<query>` (str, positional)
* `python cli/keyword_search_cli.py build`
* `python cli/keyword_search_cli.py tf <doc_id:int> <term:str>`
* `python cli/keyword_search_cli.py idf <term:str>`
* `python cli/keyword_search_cli.py tfidf <doc_id:int> <term:str>`
* `python cli/keyword_search_cli.py bm25idf <term:str>`
* `python cli/keyword_search_cli.py bm25tf <doc_id:int> <term:str> [<k1:float>] [<b:float>]`
  * `<k1>` (float, optional; default: `1.5`)
  * `<b>` (float, optional; default: `0.75`)

### `cli/semantic_search_cli.py`

Subcommands: `verify`, `verify_embeddings`, `embed_chunks`, `embed_text`, `embedquery`, `chunk`, `semantic_chunk`, `search_chunked`, `search`

* `python cli/semantic_search_cli.py verify`
* `python cli/semantic_search_cli.py verify_embeddings`
* `python cli/semantic_search_cli.py embed_chunks`
* `python cli/semantic_search_cli.py embed_text <query>`
* `python cli/semantic_search_cli.py embedquery <query>`
* `python cli/semantic_search_cli.py chunk <query> [--chunk-size <int>] [--overlap <int>]`
  * `--chunk-size` (int, default: `200`)
  * `--overlap` (int, default: `0`)
* `python cli/semantic_search_cli.py semantic_chunk <query> [--max-chunk-size <int>] [--overlap <int>]`
  * `--max-chunk-size` (int, default: `4`)
  * `--overlap` (int, default: `0`)
* `python cli/semantic_search_cli.py search_chunked <query> [--limit <int>]`
  * `--limit` (int, default: `5`)
* `python cli/semantic_search_cli.py search <query> [--limit <int>]`
  * `--limit` (int, default: `5`)

## Models Used

* Embeddings model: `sentence-transformers/all-MiniLM-L6-v2` (used by `cli/lib/semantic_search.py`)
* LLM (Gemini API): default model name `gemma-3-27b-it` (used by `cli/lib/llm.py` via `generate_response()`)
* Reranker model (cross-encoder): `cross-encoder/ms-marco-TinyBERT-L2-v2` (used by `cli/lib/llm.py` when `--rerank-method cross_encoder`)

## Environment Variables

* `HF_TOKEN`: Hugging Face token used for embedding and cross-encoder models
* `GEMINI_API_KEY`: required for LLM calls in `cli/lib/llm.py`
