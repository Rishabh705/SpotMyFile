from src.logging import logger
from src.config import Config
import os
import numpy as np
import cv2
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import face_recognition
from src.utils import create_batches

class ImageIndexer:
    """Indexes images with face recognition and object detection capabilities"""
    
    def __init__(self, feature_extractor, image_cache_manager, lock):
        """Initialize the image indexer"""
        self.feature_extractor = feature_extractor
        self.image_cache_manager = image_cache_manager
        self.lock = lock  # Thread lock for cache access
    
    def index_images(self, image_paths, max_workers=4, intent='face_recognition',
                    batch_size=Config.DEFAULT_BATCH_SIZE, max_batch_memory=100 * 1024 * 1024):
        """
        Index images by extracting features for face recognition or object detection
        based on the specified intent, using a batching approach for better memory management
        """        
        # Check that intent is valid
        if intent not in ['face_recognition', 'object_detection']:
            logger.warning(f"Invalid intent: {intent}. Using face_recognition.")
            intent = 'face_recognition'
                            
        # Filter out already processed images for this intent
        new_images = []
        for img in image_paths:
            if not self.image_cache_manager.contains(img):
                new_images.append(img)
            else:
                # Check if the specific features for this intent exist
                features = self.image_cache_manager.get_item(img)
                if features is None:
                    new_images.append(img)
                elif intent == 'face_recognition' and ('face_encodings' not in features or not features['face_encodings']):
                    new_images.append(img)
                elif intent == 'object_detection' and ('object_features' not in features or not features['object_features']):
                    new_images.append(img)
        
        if not new_images:
            logger.info(f"All images already indexed for {intent}.")
            return 0
        
        logger.info(f"Indexing {len(new_images)} new images for {intent}...")
        logger.info(f"Current image cache size: {self.image_cache_manager.get_size()} images")
        
        successful_images = 0
        
        # Create batches 
        batches = create_batches(new_images, batch_size, max_batch_memory)
        
        # Process each batch separately
        for batch_idx, batch_images in enumerate(batches):
            logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} with {len(batch_images)} images")
            
            if intent == 'face_recognition':
                # For face recognition, separate I/O from CPU operations
                successful_images += self._process_face_recognition_batch(batch_images, batch_idx, max_workers)
            else:
                # For object detection, use the original multithreaded approach
                # since object detection has thread safety issues that are handled with locks
                successful_images += self._process_object_detection_batch(batch_images, batch_idx, max_workers)
            
            # Save the cache after each batch
            with self.lock:
                self.image_cache_manager.save_cache(force=True)
                
        logger.info(f"Successfully processed {successful_images} new images for {intent}")
        logger.info(f"New image cache size: {self.image_cache_manager.get_size()} images")
        
        return successful_images
    
    def _process_face_recognition_batch(self, batch_images, batch_idx, max_workers):
        """Process a batch of images for face recognition, separating I/O and CPU operations"""
        successful_images = 0
        loaded_images = {}
        
        # Step 1: Load images in parallel (I/O bound)
        logger.info(f"Loading images for batch {batch_idx + 1}")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_image = {
                executor.submit(self._load_image, image_path): image_path
                for image_path in batch_images
            }
            
            # Process completed loading tasks with a progress bar
            for future in tqdm(
                as_completed(future_to_image),
                total=len(batch_images),
                desc=f"Loading images (batch {batch_idx + 1})"
            ):
                image_path = future_to_image[future]
                try:
                    image_data = future.result()
                    if image_data is not None:
                        loaded_images[image_path] = image_data
                except Exception as e:
                    logger.error(f"Exception loading image {image_path}: {e}")
        
        # Step 2: Process face encodings sequentially (CPU bound)
        logger.info(f"Processing face encodings for {len(loaded_images)} loaded images")
        for image_path, image_data in tqdm(
            loaded_images.items(),
            desc=f"Processing face encodings (batch {batch_idx + 1})"
        ):
            try:
                # Extract face encodings
                face_encodings = self.feature_extractor.face_model.face_encodings(image_data)
                
                # Update features in cache
                features = {}
                with self.lock:
                    existing_features = self.image_cache_manager.get_item(image_path)
                    if existing_features:
                        features = existing_features
                
                features['face_encodings'] = face_encodings
                
                # Store in cache
                with self.lock:
                    self.image_cache_manager.add_item(image_path, features)
                    # Save cache periodically
                    if self.image_cache_manager.get_size() % Config.SAVE_INTERVAL == 0:
                        self.image_cache_manager.save_cache()
                
                successful_images += 1
            except Exception as e:
                logger.error(f"Error processing face encoding for {image_path}: {e}")
        
        return successful_images
    
    def _process_object_detection_batch(self, batch_images, batch_idx, max_workers):
        """Process a batch of images for object detection using the original multithreaded approach"""
        successful_images = 0
        
        # Use the original multithreaded approach for object detection
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_image = {
                executor.submit(self._process_single_image, image_path, 'object_detection'): image_path
                for image_path in batch_images
            }
            
            # Process completed tasks with a progress bar
            for future in tqdm(
                as_completed(future_to_image),
                total=len(batch_images),
                desc=f"Extracting object features (batch {batch_idx + 1})"
            ):
                image_path = future_to_image[future]
                try:
                    result = future.result()
                    if result:
                        successful_images += 1
                except Exception as e:
                    logger.error(f"Exception extracting object features from {image_path}: {e}")
        
        return successful_images
    
    def _load_image(self, image_path):
        """Load an image from disk - I/O bound operation"""
        try:
            # Use face_recognition to load the image
            return face_recognition.load_image_file(image_path)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def _process_single_image(self, image_path, intent):
        """Process a single image to extract features based on intent - legacy method for object detection"""
        try:
            features = {}
            
            # Get existing features if any
            with self.lock:
                existing_features = self.image_cache_manager.get_item(image_path)
                if existing_features:
                    features = existing_features
            
            # Extract only the features needed for the intent
            if intent == 'face_recognition':
                face_encodings = self.feature_extractor.extract_face_features(image_path)
                features['face_encodings'] = face_encodings
            elif intent == 'object_detection':
                # Use lock only for object detection, which has the thread safety issue
                with self.lock:
                    object_features = self.feature_extractor.extract_object_features(image_path)
                features['object_features'] = object_features
            
            # Store in cache and save periodically
            with self.lock:
                self.image_cache_manager.add_item(image_path, features)
                # Save cache periodically using Config.SAVE_INTERVAL
                if self.image_cache_manager.get_size() % Config.SAVE_INTERVAL == 0:
                    self.image_cache_manager.save_cache()
            
            return True
        except Exception as e:
            logger.error(f"Error processing image {image_path} for {intent}: {e}")
            return False

class ImageSearcher:
    """Searches indexed images using face recognition and object detection"""
    
    def __init__(self, image_cache_manager, feature_extractor):
        """Initialize the image searcher"""
        self.image_cache_manager = image_cache_manager
        self.feature_extractor = feature_extractor
    
    def search(self, query, intent, top_k=5):
        """
        Search indexed images based on intent:
        - 'face_recognition': Searches for similar faces to the query image
        - 'object_detection': Searches for images containing the object specified in query
        """
        cache_size = self.image_cache_manager.get_size()
        logger.info(f"Image cache size before search: {cache_size} images")
        
        if cache_size == 0:
            logger.warning("No images have been indexed yet. Please index images first.")
            return []
        
        if intent == 'face_recognition':
            return self._search_faces(query, top_k)
        elif intent == 'object_detection':
            return self._search_objects(query, top_k)
        else:
            logger.warning(f"Unknown search intent: {intent}")
            return []
    
    def _search_faces(self, query_image_path, top_k=5):
        """Search for images containing faces similar to those in the query image"""
        logger.info(f"Searching for faces similar to those in {query_image_path}")
        
        # Extract face encodings from query image
        query_encodings = self.feature_extractor.extract_face_features(query_image_path)
        
        if not query_encodings:
            logger.warning("No faces found in the query image.")
            return []
        
        # Get all image paths from cache
        all_images = self.image_cache_manager.get_all_keys()
        
        # Compare face encodings
        results = []
        
        for image_path in all_images:
            features = self.image_cache_manager.get_item(image_path)
            if features and 'face_encodings' in features and features['face_encodings']:
                # For each face in the cached image
                for face_encoding in features['face_encodings']:
                    # Compare with each face in the query image
                    for query_encoding in query_encodings:
                        # Use face_recognition library to compare faces
                        face_distance = face_recognition.face_distance([face_encoding], query_encoding)[0]
                        # Convert distance to similarity score (lower distance means higher similarity)
                        similarity_score = 1 - min(face_distance, 1.0)
                        
                        # Use the threshold from Config
                        if similarity_score > Config.FACE_SIMILARITY_THRESHOLD:
                            results.append((image_path, similarity_score))
                            break  # No need to check other faces in this image
                    
                    # If we already found a match in this image, move to the next image
                    if results and results[-1][0] == image_path:
                        break
        
        # Sort by similarity score (higher is better)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        return results[:top_k]
    
    def _search_objects(self, object_name, top_k=5):
        """Search for images containing the specified object"""
        logger.info(f"Searching for images containing object: {object_name}")
        
        # Get the class ID for the object name
        try:
            if not hasattr(self.feature_extractor, 'classes'):
                logger.warning("No class list available in feature extractor.")
                return []
                
            class_id = self.feature_extractor.classes.index(object_name.lower())
        except ValueError:
            logger.warning(f"Object '{object_name}' not found in the known classes.")
            return []
        
        # Get all image paths from cache
        all_images = self.image_cache_manager.get_all_keys()
        
        # Search for images containing the object
        results = []
        
        for image_path in all_images:
            features = self.image_cache_manager.get_item(image_path)
            if features and 'object_features' in features and features['object_features']:
                obj_features = features['object_features']
                if 'class_ids' in obj_features and class_id in obj_features['class_ids']:
                    # Find all instances of this object in the image
                    indices = [i for i, c_id in enumerate(obj_features['class_ids']) if c_id == class_id]
                    
                    if indices:
                        # Use the highest confidence score for this object
                        confidence = max([obj_features['confidences'][i] for i in indices])
                        results.append((image_path, confidence))
        
        # Sort by confidence score (higher is better)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        return results[:top_k]