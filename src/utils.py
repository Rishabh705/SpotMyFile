from src.logging import logger
from src.config import Config
from typing import Any, Dict, List, Optional, Set, Tuple
from pathlib import Path
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
class FileSystem:
    """Utilities for working with the file system"""
    
    @staticmethod
    def get_files_in_directory(
        directory: str,
        extensions: Set[str] = None,
        exclude_keywords: List[str] = None,
        exclude_filenames: List[str] = None
    ) -> List[str]:
        """
        Get all files in a directory matching the criteria
        """
        if extensions is None:
            extensions = set(Config.FILE_EXTENSIONS)
        if exclude_keywords is None:
            exclude_keywords = Config.DEFAULT_EXCLUDE_KEYWORDS
        if exclude_filenames is None:
            exclude_filenames = Config.DEFAULT_EXCLUDE_FILENAMES
        
        # Normalize extensions to lowercase
        extensions = {ext.lower() for ext in extensions}
        
        all_files = [
            str(path) for path in Path(directory).rglob("*")
            if path.suffix.lower() in extensions
        ]
        
        filtered_files = [
            f for f in all_files
            if not any(ex_kw.lower() in f.lower() for ex_kw in exclude_keywords)
            and not any(Path(f).name.lower() == ex_fn.lower() for ex_fn in exclude_filenames)
        ]
        
        return filtered_files

def create_batches(
    file_paths: List[str],
    batch_size: int,
    max_batch_memory: int
) -> List[List[str]]:
    """
    Create batches of file paths based on file size and memory constraints.
    """
    batches = []
    current_batch = []
    current_batch_size = 0

    for file_path in file_paths:
        # Estimate file size
        file_size = os.path.getsize(file_path)

        # If adding this file would exceed our memory target, start a new batch
        if len(current_batch) >= batch_size or current_batch_size + file_size > max_batch_memory:
            if current_batch:  # Don't add empty batches
                batches.append(current_batch)
            current_batch = [file_path]
            current_batch_size = file_size
        else:
            current_batch.append(file_path)
            current_batch_size += file_size

    # Add the final batch if not empty
    if current_batch:
        batches.append(current_batch)

    return batches


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_image_with_boxes(filename, features, folder_path, classes):
    """
    Plot an image with bounding boxes around detected objects.
    
    """
    img_path = os.path.join(folder_path, filename)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    
    for feature in features:
        bbox = feature['bounding_box']
        class_id = feature['class_id']
        confidence = feature['confidence']
        x, y, w, h = bbox
        
        # Draw rectangle and label
        plt.gca().add_patch(plt.Rectangle((x, y), w, h, edgecolor='red', linewidth=2, fill=False))
        plt.text(x, y-10, f"{classes[class_id]}: {confidence:.2f}", 
                 bbox=dict(facecolor='red', alpha=0.5), color='white')
    
    plt.title(filename)
    plt.show()

def plot_matching_images(matching_images, intent, folder_path, input_image_path=None):
    """
    Plot matching images found during search.
    """
    if not matching_images:
        print("No matching images to display.")
        return
        
    num_images = len(matching_images)
    if intent == 'text_image':
        # Extract just the filenames for text_image intent
        matching_images = [img for img, _ in matching_images]
        num_images = len(matching_images)
    
    plt.figure(figsize=(15, 5 * (1 + (num_images // 3))))
    
    # For face intent, display the input image as well
    plot_offset = 0
    if intent == 'face' and input_image_path:
        plt.subplot(1 + (num_images // 3), 3, 1)
        input_image = Image.open(input_image_path)
        plt.imshow(np.array(input_image))
        plt.axis('off')
        plt.title("Input Image")
        plot_offset = 1
    
    # Plot matching images
    for i, result in enumerate(matching_images):
        plt.subplot(1 + (num_images // 3), 3, i + 1 + plot_offset)
        
        if intent == 'text_image':
            img_path = os.path.join(folder_path, result)
        else:
            img_path = os.path.join(folder_path, result)
            
        img = Image.open(img_path)
        plt.imshow(np.array(img))
        plt.axis('off')
        
        if intent == 'text_image':
            plt.title(f"Match {i + 1}")
        else:
            plt.title(f"Match {i + 1}")
            
    plt.tight_layout()
    plt.show()