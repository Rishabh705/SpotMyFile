from src.config import Config
from src.text import TextProcessor
from src.file import FileIndexer, FileSearcher
from src.cache import CacheManager
from src.logging import logger
from src.utils import FileSystem
import time
import multiprocessing as mp
import os
# Environment variables
import dotenv
dotenv.load_dotenv()

class Scout:
    """Main application class for Scout search engine"""
    
    def __init__(self, model_name=None):
        """Initialize Scout with configuration"""
        # Initialize configuration
        Config.initialize()
        
        # Set up components
        self.text_processor = TextProcessor(model_name)
        self.embeddings_cache = CacheManager(Config.get_cache_path('text'))
        self.content_cache = CacheManager(Config.get_cache_path('content'))
        
        # Create indexer and searcher
        self.indexer = FileIndexer(self.text_processor, self.embeddings_cache, self.content_cache)
        self.searcher = FileSearcher(self.text_processor, self.embeddings_cache, self.content_cache)
        
        # Track if BM25 is initialized
        self.bm25_initialized = False
    
    def index_directory(self, directory_path, extensions=None):
        """Index all files in a directory"""
        logger.info(f"Scanning for files in {directory_path}...")
        file_paths = FileSystem.get_files_in_directory(
            directory_path,
            extensions=extensions or Config.FILE_EXTENSIONS,
            exclude_keywords=Config.DEFAULT_EXCLUDE_KEYWORDS,
            exclude_filenames=Config.DEFAULT_EXCLUDE_FILENAMES
        )
        
        logger.info(f"Found {len(file_paths)} files to process")
        
        # Show sample files
        if file_paths:
            logger.info("Sample files:")
            for i, path in enumerate(file_paths[:5]):
                logger.info(f"  {i+1}. {path}")
            if len(file_paths) > 5:
                logger.info(f"  ... and {len(file_paths) - 5} more")

            default_workers = max(1, mp.cpu_count() // 2)
            user_input = input(f"Enter number of workers (default {default_workers}): ")
            workers = int(user_input) if user_input.strip() else default_workers
            
            # Index files
            start_time = time.time()
            processed_count = self.indexer.index_files(
                file_paths, 
                max_workers=workers,
                batch_size=Config.DEFAULT_BATCH_SIZE,
                timeout=Config.DEFAULT_TIMEOUT
            )
            end_time = time.time()
            
            logger.info(f"Indexing completed in {end_time - start_time:.2f} seconds")
            
            # Initialize BM25 if needed
            print(f"[DEBUG] Content cache size: {self.content_cache.get_size()}")
            if self.content_cache.get_size() > 0:
                self.bm25_initialized = self.searcher.initialize_bm25()
        else:
            logger.warning("No files to index.")
    
    def search(self, query, top_k=5, include_preview=True, hybrid=True, vector_weight=0.7):
        """Search for files matching the query"""
        logger.info(f"Searching for: '{query}'...")
        
        if hybrid and self.bm25_initialized:
            logger.info(f"Using hybrid search (vector_weight={vector_weight})")
            return self.searcher.hybrid_search(
                query, 
                top_k=top_k, 
                vector_weight=vector_weight,
                include_preview=include_preview
            )
        else:
            if hybrid and not self.bm25_initialized:
                logger.info("BM25 not initialized, falling back to vector search")
            return self.searcher.search(query, top_k=top_k, include_preview=include_preview)

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================


if __name__ == "__main__":
    
    """Main entry point for the Scout application"""
    print("\n🔍 SCOUT - Universal Search Engine for PC\n")
    
    # Text mode is the only implemented mode
    intent = 'file'
    
    if intent == 'file':
        # Get folder path
        default_path = os.getenv("FOLDER_PATH")
        folder_input = input(f"Enter folder path to index (default: {default_path}): ").strip()
        folder_path = folder_input if folder_input else default_path
        print(f"Using folder path: {folder_path}")
        
        # Select embedding model
        names = [name for name in Config.MODEL_ALIASES]
        model_choice = input(f"Choose model {names}: ").strip().lower()
        model_name = Config.MODEL_ALIASES.get(model_choice, Config.DEFAULT_MODEL)
        print(f"Using model: {model_name}")
        
        # Initialize Scout
        scout = Scout(model_name)
        
        # Index directory
        scout.index_directory(folder_path)
        
        # Search loop
        while True:
            query = input("\nEnter your text query (or 'q' to quit): ")
            if query.lower() in ('q', 'quit', 'exit'):
                break
            
            # Ask for search mode
            search_mode = input("Search mode (hybrid/vector) [default: hybrid]: ").strip().lower()
            use_hybrid = search_mode != "vector"  # Default to hybrid
            
            if use_hybrid:
                weight_input = input("Vector weight (0.0-1.0) [default: 0.7]: ").strip()
                try:
                    vector_weight = float(weight_input) if weight_input else 0.7
                    # Ensure weight is in valid range
                    vector_weight = max(0.0, min(1.0, vector_weight))
                except ValueError:
                    vector_weight = 0.7
                    print("Invalid weight, using default 0.7")
            else:
                vector_weight = 1.0  # Pure vector search
                
            top_k_input = input("Number of results to show [default: 5]: ").strip()
            try:
                top_k = int(top_k_input) if top_k_input else 5
            except ValueError:
                top_k = 5
                print("Invalid number, showing 5 results")
                
            results = scout.search(
                query, 
                top_k=top_k, 
                include_preview=True, 
                hybrid=use_hybrid,
                vector_weight=vector_weight
            )
            
            print("\nSearch Results:")
            if results:
                for i, (file_path, similarity, preview) in enumerate(results):
                    print(f"{i+1}. File: {file_path}")
                    print(f"   Similarity: {similarity:.4f}")
                    
                    # Format preview text nicely
                    if preview:
                        # Wrap long preview text
                        import textwrap
                        wrapped_preview = textwrap.fill(preview[:500], width=80, initial_indent="   ", subsequent_indent="      ")
                        print(f"   Preview:")
                        print(wrapped_preview)
                        if len(preview) > 500:
                            print("      ...")
                    else:
                        print("   Preview: Not available")
                    print()
            else:
                print("No matching files found.")
        print("Goodbye...")