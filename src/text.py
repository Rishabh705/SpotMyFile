# src/text.py
from src.config import Config
from src.logging import logger
import re
from textblob import TextBlob
from sentence_transformers import SentenceTransformer
import numpy as np
import os

class TextProcessor:
    """Text processing and embedding utilities"""
    
    def __init__(self, model_name=None):
        """Initialize text processor with a specific model"""
        model_name = model_name or Config.DEFAULT_MODEL
        if model_name in Config.MODEL_ALIASES:
            model_name = Config.MODEL_ALIASES[model_name]
        self.model_name = model_name
        self._model = None  # Lazy-loaded
    
    @property
    def model(self):
        """Lazy-load the embedding model"""
        if self._model is None:
            logger.info(f"Loading text embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    @staticmethod
    def clean_text(text):
        """Clean text by removing extra whitespace and normalizing"""
        if not isinstance(text, str):
            return ""
        # Remove extra whitespace and normalize
        return re.sub(r'\s+', ' ', text.strip().lower())
    
    @staticmethod
    def correct_spelling(text):
        """Attempt to correct spelling in the text using TextBlob"""
        try:
            return str(TextBlob(text).correct())
        except Exception as e:
            logger.error(f"Error correcting spelling: {e}")
            return text
    
    def preprocess_query(self, query):
        """Preprocess query for better model performance"""
        if "bge" in self.model_name.lower():
            return f"Represent this sentence for searching relevant passages: {query}"
        return query
        
    def encode_text(self, text_or_texts):
        """Create normalized embedding(s) for input text(s) or list of texts with memory optimization"""
        # Maximum text length to process at once
        MAX_TEXT_LENGTH = 1000  # Adjust based on your model and memory constraints
        
        # Handle single text case
        if isinstance(text_or_texts, str):
            # Apply query preprocessing if it's a single query
            if "bge" in self.model_name.lower():
                text_or_texts = self.preprocess_query(text_or_texts)
                
            cleaned = self.clean_text(text_or_texts)
            if not cleaned:
                return np.zeros(self.model.get_sentence_embedding_dimension())
            texts = [cleaned]
            single_input = True
        else:
            # Filter out empty texts after cleaning
            texts = [self.clean_text(t) for t in text_or_texts if t]
            texts = [t for t in texts if t]  # Remove any that became empty after cleaning
            single_input = False
            
            if not texts:
                # Return appropriate zeros if all texts were empty
                zero = np.zeros(self.model.get_sentence_embedding_dimension())
                return zero if single_input else [zero] * len(text_or_texts)
        
        try:
            # Truncate texts to prevent memory issues
            truncated_texts = [text[:MAX_TEXT_LENGTH] for text in texts]
            
            # Process in smaller batches to reduce memory usage
            results = []
            
            # A more conservative batch size
            max_batch_size = 16  # Smaller batch size to prevent memory issues
            
            for i in range(0, len(truncated_texts), max_batch_size):
                batch = truncated_texts[i:i+max_batch_size]
                try:
                    # Use a smaller batch_size for the model encode call
                    batch_embeddings = self.model.encode(batch, show_progress_bar=False, batch_size=4)
                    results.extend(batch_embeddings)
                except Exception as e:
                    logger.error(f"Error encoding batch {i//max_batch_size}: {e}")
                    # Fill with zeros for failed batch
                    zeros = [np.zeros(self.model.get_sentence_embedding_dimension()) for _ in range(len(batch))]
                    results.extend(zeros)
            
            embeddings = np.array(results)
            
            # Normalize embeddings (L2 norm)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            # Avoid division by zero
            norms[norms < 1e-10] = 1.0
            embeddings = embeddings / norms
            
            # Return appropriate format based on input type
            return embeddings[0] if single_input else embeddings
        except Exception as e:
            logger.error(f"Error encoding text(s): {e}")
            zero = np.zeros(self.model.get_sentence_embedding_dimension())
            return zero if single_input else [zero] * len(text_or_texts)
