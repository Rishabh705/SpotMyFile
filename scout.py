import os
import time
import threading
from src.config import Config
from src.utils import FileSystem
from dotenv import load_dotenv
load_dotenv()

# Early initialization of Config
Config.initialize()

# Create a simple console progress indicator
def print_progress(message):
    print(f"⏳ {message}")

class Scout:
    """Main application class for Scout search engine"""
    
    def __init__(self, model_name=None, intent=None):
        """Initialize Scout with only components needed for the specified intent"""
        # Add a lock for cache access
        self.cache_lock = threading.Lock()
        
        # Initialize base cache managers
        self.embeddings_cache = None
        self.content_cache = None
        self.face_cache = None
        self.object_cache = None
        
        # Lazily load only the components needed for the specified intent
        if intent == 'text':
            self._initialize_text_components(model_name)
        elif intent == 'face_recognition':
            self._initialize_face_components()
        elif intent == 'object_detection':
            self._initialize_object_components()
        
        # Track if BM25 is initialized
        self.bm25_initialized = False
    
    def _initialize_text_components(self, model_name):
        """Initialize only text-related components"""
        print_progress("Loading text processing components...")
        
        # Import only when needed
        from src.text import TextProcessor
        from src.file import FileIndexer, FileSearcher
        from src.cache import CacheManager
        from src.extractor import TextContentExtractor
        
        # Initialize text components
        self.text_processor = TextProcessor(model_name)
        self.embeddings_cache = CacheManager(Config.get_cache_path('text'))
        self.content_cache = CacheManager(Config.get_cache_path('content'))
        self.content_extractor = TextContentExtractor()
        self.text_indexer = FileIndexer(self.text_processor, self.embeddings_cache, 
                                      self.content_cache, self.content_extractor, 
                                      self.cache_lock)
        self.text_searcher = FileSearcher(self.text_processor, self.embeddings_cache, 
                                        self.content_cache)
        print_progress("Text components loaded.")
    
    def _initialize_face_components(self):
        """Initialize only face recognition components"""
        print_progress("Loading face recognition components...")
        
        # Import only when needed
        from src.cache import CacheManager
        from src.extractor import ImageFeatureExtractor
        from src.image import ImageIndexer, ImageSearcher
        
        # Initialize face components
        self.face_cache = CacheManager(Config.get_cache_path('face'))
        self.feature_extractor = ImageFeatureExtractor()
        self.face_indexer = ImageIndexer(self.feature_extractor, self.face_cache, self.cache_lock)
        self.face_searcher = ImageSearcher(self.face_cache, self.feature_extractor)
        print_progress("Face components loaded.")
    
    def _initialize_object_components(self):
        """Initialize only object detection components"""
        print_progress("Loading object detection components...")
        
        # Import only when needed
        from src.cache import CacheManager
        from src.extractor import ImageFeatureExtractor
        from src.image import ImageIndexer, ImageSearcher
        
        # Initialize object components
        self.object_cache = CacheManager(Config.get_cache_path('object'))
        self.feature_extractor = ImageFeatureExtractor()
        self.object_indexer = ImageIndexer(self.feature_extractor, self.object_cache, self.cache_lock)
        self.object_searcher = ImageSearcher(self.object_cache, self.feature_extractor)
        print_progress("Object detection components loaded.")
    
    def index_directory(self, directory_path, intent):
        """Index all relevant files in a directory"""
        from src.logging import logger
        logger.info(f"Scanning for files in {directory_path}...")
        
        # Configuration based on intent
        if intent == 1:  # Text indexing
            extensions = Config.FILE_EXTENSIONS
            indexer = self.text_indexer
            file_type = "text"
            indexer_intent = None  # Not used for text indexing
        elif intent == 2:  # Face recognition
            extensions = Config.IMAGE_EXTENSIONS
            indexer = self.face_indexer  # Use face-specific indexer
            file_type = "images"
            indexer_intent = 'face_recognition'
        elif intent == 3:  # Object detection
            extensions = Config.IMAGE_EXTENSIONS
            indexer = self.object_indexer  # Use object-specific indexer
            file_type = "images"
            indexer_intent = 'object_detection'
        else:
            logger.warning(f"Unknown intent: {intent}")
            return
        
        # Common file scanning logic
        paths = FileSystem.get_files_in_directory(
            directory_path,
            extensions=extensions,
            exclude_keywords=Config.DEFAULT_EXCLUDE_KEYWORDS,
            exclude_filenames=Config.DEFAULT_EXCLUDE_FILENAMES
        )
        
        logger.info(f"Found {len(paths)} {file_type} to process")
        
        if not paths:
            logger.warning(f"No {file_type} to index.")
            return
            
        # Log sample files
        logger.info(f"Sample {file_type}:")
        for i, path in enumerate(paths[:5]):
            logger.info(f"  {i+1}. {path}")
        if len(paths) > 5:
            logger.info(f"  ... and {len(paths) - 5} more")
        
        # Get worker count
        import multiprocessing as mp
        default_workers = max(1, mp.cpu_count() // 2)
        worker_prompt = f"Enter number of {file_type} workers (default {default_workers}): "
        user_input = input(worker_prompt)
        workers = int(user_input) if user_input.strip() else default_workers
        
        # Perform indexing with timing
        start_time = time.time()
        
        # Use dynamic method calling with the appropriate arguments
        if intent == 1:  # Text indexing requires additional parameters
            processed_count = indexer.index_files(
                paths, 
                max_workers=workers,
                batch_size=Config.DEFAULT_BATCH_SIZE,
                timeout=Config.DEFAULT_TIMEOUT
            )
        else:  # Image indexing with specific intent
            processed_count = indexer.index_images(
                paths, 
                max_workers=workers,
                intent=indexer_intent
            )
        
        end_time = time.time()
        
        logger.info(f"{file_type.capitalize()} indexing completed in {end_time - start_time:.2f} seconds")
        
        # Additional work for text indexing
        if intent == 1 and self.content_cache.get_size() > 0:
            self.bm25_initialized = self.text_searcher.initialize_bm25()

    def search(self, query, top_k=5, include_preview=True, hybrid=True, vector_weight=0.7, intent=None, folder_path=None):
        """Search for files or images matching the query based on intent"""
        from src.logging import logger
        logger.info(f"Searching for: '{query}' with intent: {intent}...")
        
        # Get search results based on intent
        results = []
        if intent == 'text':
            if hybrid and self.bm25_initialized:
                logger.info(f"Using hybrid search (vector_weight={vector_weight})")
                results = self.text_searcher.hybrid_search(
                    query, 
                    top_k=top_k*2,  # Get more results than needed to allow for filtering
                    vector_weight=vector_weight,
                    include_preview=include_preview
                )
            else:
                if hybrid and not self.bm25_initialized:
                    logger.info("BM25 not initialized, falling back to vector search")
                results = self.text_searcher.search(query, top_k=top_k*2, include_preview=include_preview)
        elif intent == 'face_recognition':
            results = self.face_searcher.search(query, intent, top_k=top_k*2)
        elif intent == 'object_detection':
            results = self.object_searcher.search(query, intent, top_k=top_k*2)
        else:
            logger.warning(f"Unknown intent: {intent}")
            return []
        
        # Filter results to only include files from the specified folder (if provided)
        if folder_path:
            folder_path = os.path.normpath(folder_path)
            filtered_results = []
            for result in results:
                file_path = result[0]  # First element is the file path
                if os.path.normpath(file_path).startswith(folder_path):
                    filtered_results.append(result)
                if len(filtered_results) >= top_k:
                    break
            return filtered_results[:top_k]
        return results[:top_k]
        
# Factory functions to create specific Scout instances
def create_text_scout(model_name=None):
    """Create a Scout instance for text search"""
    return Scout(model_name=model_name, intent='text')

def create_face_scout():
    """Create a Scout instance for face recognition"""
    return Scout(intent='face_recognition')

def create_object_scout():
    """Create a Scout instance for object detection"""
    return Scout(intent='object_detection')

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    """Main entry point for the Scout application"""
    print("\n🔍 SCOUT - File Search Tool for PC\n")
    print("Preparing basic components...")
    
    # Ensure Config is initialized but don't create Scout yet
    Config.initialize()
    
    # Make sure logging directories exist
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    
    # Import logger after directories are created
    from src.logging import logger
    
    while True:  # Main program loop
        print("\nSelect intent:")
        print("1. Text Search")
        print("2. Face Recognition")
        print("3. Object Detection")
        print("4. Exit Program")

        intent = input("Enter intent number (1, 2, 3, or 4): ").strip()
        
        if intent == '4':
            print("Exiting program...")
            break
            
        # Get folder path - moved outside the intent blocks
        default_path = os.getenv("FOLDER_PATH")
        folder_input = input(f"Enter folder path to index (default: {default_path}): ").strip()
        folder_path = folder_input if folder_input else default_path
        print(f"Using folder path: {folder_path}")
        
        # Create appropriate Scout instance based on intent
        scout = None
        
        if intent == '1':
            # Select embedding model
            names = [name for name in Config.MODEL_ALIASES]
            model_choice = input(f"Choose model {names}: ").strip().lower()
            model_name = Config.MODEL_ALIASES.get(model_choice, Config.DEFAULT_MODEL)
            print(f"Using model: {model_name}")
            
            # Initialize text-specific Scout
            scout = create_text_scout(model_name)
            
            # Index directory
            scout.index_directory(folder_path, 1)
            
            # Search loop
            while True:
                query = input("\nEnter your text query (or 'q' to return to main menu): ")
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
                    vector_weight=vector_weight,
                    intent='text',
                    folder_path=folder_path
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
                    
            print("Returning to intent selection...")

        elif intent == '2':
            # Initialize face-specific Scout
            scout = create_face_scout()

            # Index directory
            scout.index_directory(folder_path, 2)  # Using 2 for face recognition
            
            while True:
                # Query Image Path
                query = input("\nEnter query image path (or 'q' to return to main menu): ").strip()
                if query.lower() in ('q', 'quit', 'exit'):
                    break
                    
                top_k_input = input("Number of results to show [default: 5]: ").strip()
                try:
                    top_k = int(top_k_input) if top_k_input else 5
                except ValueError:
                    top_k = 5
                    print("Invalid number, showing 5 results")

                results = scout.search(query, top_k=top_k, intent='face_recognition', folder_path=folder_input)

                print("\nSearch Results:")
                if results:
                    for i, (image_path, score) in enumerate(results):
                        print(f"{i+1}. Image: {image_path}, Score: {score}")
                        
                    # Import visualization only when needed
                    from src.utils import plot_matching_images
                    # Use the visualization function
                    matching_images = [path for path, _ in results]
                    plot_matching_images(matching_images, 'face', folder_path, input_image_path=query)
                else:
                    print("No matching images found.")
                    
            print("Returning to intent selection...")

        elif intent == '3':
            # Initialize object-specific Scout
            scout = create_object_scout()

            # Index directory
            scout.index_directory(folder_path, 3)  # Using 3 for object detection
            
            while True:
                # Target object
                query = input("\nEnter target object (or 'q' to return to main menu): ").strip()
                if query.lower() in ('q', 'quit', 'exit'):
                    break

                top_k_input = input("Number of results to show [default: 5]: ").strip()
                try:
                    top_k = int(top_k_input) if top_k_input else 5
                except ValueError:
                    top_k = 5
                    print("Invalid number, showing 5 results")

                results = scout.search(query, top_k=top_k, intent='object_detection', folder_path=folder_input)

                if results:
                    print("\nSearch Results:")
                    for i, (image_path, score) in enumerate(results):
                        print(f"{i+1}. Image: {image_path}, Score: {score}")
                        
                    # Import visualization only when needed
                    from src.utils import plot_matching_images
                    # Use the visualization function
                    matching_images = [path for path, _ in results]
                    plot_matching_images(matching_images, 'object', folder_path)
                else:
                    print("No matching images found.")
                    
            print("Returning to intent selection...")
        else:
            print("Invalid intent selection. Please try again.")
            
    print("Goodbye!")