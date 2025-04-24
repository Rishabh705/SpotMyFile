# src/text.py
from src.config import Config
from src.logging import logger
import re
from textblob import TextBlob
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import docx
import pytesseract
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import numpy as np
import os
import concurrent.futures

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


class ContentExtractor:
    """Extracts text content from various file types"""
    
    @staticmethod
    def extract_text_from_txt(file_path):
        """
        Extract text from a TXT file.
        """
        try:
            with open(file_path, 'r', encoding="utf-8", errors="replace") as file:
                return file.read()[:Config.MAX_CHARS]
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return ""
    
    @staticmethod
    def process_page(file_path, page_num, ocr_dpi, tesseract_config):
        """Process a single PDF page with text extraction and optional OCR"""
        try:
            with fitz.open(file_path) as doc:
                page = doc.load_page(page_num)
                text = page.get_text("text")
                if not text.strip():  # If no text found, try OCR
                    pix = page.get_pixmap(dpi=ocr_dpi)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    img = img.convert("L")  # Convert to grayscale to reduce memory
                    text = pytesseract.image_to_string(img, config=tesseract_config)
                return (page_num, text)
        except Exception as e:
            return (page_num, f"[ERROR on page {page_num}]: {str(e)}")
    
    @classmethod
    def extract_text_from_pdf(cls, file_path: str, ocr_dpi: int = Config.OCR_DPI, 
                            tesseract_config: str = Config.TESSERACT_CONFIG) -> str:
        """
        Extract text from a PDF file with optimized processing and memory usage
        """
        try:
            with fitz.open(file_path) as doc:
                page_count = doc.page_count
                
                # Process in smaller chunks to reduce memory usage
                chunk_size = min(5, page_count)  # Process 5 pages at a time at most
                results = []
                total_chars = 0
                
                # Use at most half of CPU cores for PDF extraction to avoid memory issues
                max_workers = max(1, os.cpu_count() // 2)
                
                for start_page in range(0, page_count, chunk_size):
                    end_page = min(start_page + chunk_size, page_count)
                    logger.debug(f"Processing PDF pages {start_page+1}-{end_page} of {page_count}")
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = [
                            executor.submit(cls.process_page, file_path, i, ocr_dpi, tesseract_config)
                            for i in range(start_page, end_page)
                        ]
                        
                        for future in concurrent.futures.as_completed(futures):
                            page_num, text = future.result()
                            remaining = Config.MAX_CHARS - total_chars
                            if remaining <= 0:
                                break
                            clipped_text = text[:remaining]
                            total_chars += len(clipped_text)
                            results.append((page_num, clipped_text))
                    
                    # Break early if reached character limit
                    if total_chars >= Config.MAX_CHARS:
                        break
                
                results.sort(key=lambda x: x[0])
                return "\n".join(text for _, text in results).strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_docx(file_path):
        """
        Extract text from a DOCX file.
        """
        try:
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text[:Config.MAX_CHARS]
        except Exception as e:
            logger.error(f"Error reading DOCX {file_path}: {e}")
            return ""
    
    @classmethod
    def extract_text(cls, file_path):
        try:
            file_extension = Path(file_path).suffix.lower()
            logger.debug(f"Attempting to extract text from {file_path} with extension {file_extension}")
            
            result = ""
            if file_extension == ".txt":
                result = cls.extract_text_from_txt(file_path)
            elif file_extension == ".pdf":
                result = cls.extract_text_from_pdf(file_path)
            elif file_extension == ".docx":
                result = cls.extract_text_from_docx(file_path)
            else:
                logger.warning(f"Unsupported file extension: {file_extension}")
                return ""
            
            if not result.strip():
                logger.warning(f"Extraction returned empty content for {file_path}")
            else:
                logger.debug(f"Successfully extracted {len(result)} characters from {file_path}")
            
            return result
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return ""