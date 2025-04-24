from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from tqdm import tqdm
from src.logging import logger

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class BM25Searcher:
    """BM25 keyword search for hybrid retrieval"""
    
    def __init__(self, content_cache):
        """Initialize with content cache"""
        self.content_cache = content_cache
        self.tokenized_corpus = {}
        self.bm25 = None
        self.file_paths = []
        
    def tokenize_text(self, text):
        """Tokenize text for BM25"""
        if not isinstance(text, str):
            return []
        return word_tokenize(text.lower())
    
    def build_index(self):
        """Build BM25 index from content cache"""
        logger.info("Building BM25 index...")
        self.file_paths = list(self.content_cache.get_all_keys())
        tokenized_corpus = []
        
        for file_path in tqdm(self.file_paths, desc="BM25 Indexing"):
            content = self.content_cache.get_item(file_path)
            if content:
                tokens = self.tokenize_text(content)
                tokenized_corpus.append(tokens)
            else:
                tokenized_corpus.append([])
                
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 index built for {len(self.file_paths)} documents")
        
    def search(self, query, top_k=5):
        """Search using BM25 algorithm"""
        if not self.bm25:
            logger.warning("BM25 index not built yet")
            return []
            
        query_tokens = self.tokenize_text(query)
        if not query_tokens:
            logger.warning("Empty query tokens after tokenization")
            return []
            
        doc_scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k results
        top_indices = np.argsort(doc_scores)[-top_k:][::-1]
        
        # Return (file_path, score) tuples
        return [(self.file_paths[idx], doc_scores[idx]) for idx in top_indices]