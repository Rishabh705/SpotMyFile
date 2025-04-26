# FileScout

Scout is a powerful search tool that combines text embedding search and image recognition capabilities. It enables users to search for both files and images based on content, with support for face recognition, object detection, and text similarity.

## Features

### File Search
- **Text Embedding Search**: Index and search files using text embeddings for semantic similarity
- **Hybrid Search**: Combine vector embeddings with BM25 lexical search for improved results
- **Adaptive Batching**: Process files in batches optimized for memory usage
- **Caching**: Features are cached to optimize repeated searches and reduce processing time
- **Multi-threaded Processing**: Efficiently process files with parallel extraction and embedding

### Image Recognition
- **Face Recognition**: Identify and match faces in images using a pre-trained face recognition model
- **Object Detection**: Utilize YOLOv4 for real-time object detection in images
- **Batch Processing**: Process images in memory-efficient batches
- **Feature Extraction**: Extract and cache image features for quick retrieval
- **Thread-safe Operations**: Safely process images in multi-threaded environments

## Architecture

The system is built with a modular architecture:

- **Indexers**: `FileIndexer` and `ImageIndexer` process and store features
- **Searchers**: `FileSearcher` and `ImageSearcher` retrieve relevant content
- **Cache Manager**: Store embeddings and features for efficient retrieval
- **Extractors**: Generate embeddings and features from content

## Requirements

- Python 3.x
- OpenCV
- NumPy
- scikit-learn
- tqdm
- face_recognition
- YOLOv4 configuration and weights

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Rishabh705/image-scout.git
   cd image-scout
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the YOLOv4 weights, configuration, and COCO names files:

   - [yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights)
   - [yolov4.cfg](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg?raw=true)
   - [coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true)

4. Configure your environment variables in a `.env` file:

   ```env
   FOLDER_PATH=path/to/your/content
   ```

## Usage


```bash
python scout.py
```
```
🔍 SCOUT - File Search Tool for PC

Preparing basic components...

Select intent:
1. Text Search
2. Face Recognition
3. Object Detection
4. Exit Program
Enter intent number (1, 2, 3, or 4): 
```
## Project Structure

```
image-scout/
├── src/
│   ├── __init__.py
│   ├── config.py               # Configuration settings
│   ├── logging.py              # Logging setup
│   ├── utils.py                # Utility functions
│   ├── file.py                 # File search implementation
│   ├── image.py                # Image search implementation
│   ├── text.py                 # Text processing and embeddings
│   ├── cache.py                # Cache management
│   ├── extractor.py            # Content extraction from files
│   └── bm25.py                 # BM25 search implementation
├── models/                     # Pre-trained models
│   ├── yolov4.weights
│   ├── yolov4.cfg
│   └── coco.names
├── caches/                      # Cache storage
│   ├── text_featues_cache.pkl
│   ├── content_cache.pkl
│   ├── object_features_cache.pklface
│   └── _features_cache.pkl
└── scout.py
```

## Performance Considerations

- **Memory Usage**: The system uses batching to manage memory usage when processing large numbers of files or images
- **Thread Count**: Adjust `max_workers` based on your system's resources and the I/O vs CPU characteristics of your workload

## Contributions
Contributions are welcome! If you'd like to improve this project, fix bugs, or add new features, feel free to fork the repository and submit a pull request.

## License

[GNU License](LICENSE)
