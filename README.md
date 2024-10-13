# ImageScout

ImageScout is an image recognition tool that utilizes YOLOv4 (You Only Look Once) for object detection and face recognition. It allows users to search for images based on specific object queries and visualize the results with bounding boxes highlighting detected objects.


## Features

- **Object Detection**: Utilizes YOLOv4 for real-time object detection in images.
- **Image Search**: Search for images in a specified folder based on user-defined queries.
- **Face Recognition**: Identify and match faces in images using a pre-trained face recognition model.
- **Caching**: Features are cached to optimize repeated searches and reduce processing time.
- **Visualization**: Display images with bounding boxes around detected objects and visual results for face recognition.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Matplotlib
- Pillow
- Python-dotenv
- face-recognition

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/imagescout.git
   cd imagescout
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the YOLOv4 weights, configuration, and COCO names files. You can obtain them from the official YOLO website or GitHub repository.

4. Create a `.env` file in the project directory with the following content:

   ```env
   FOLDER_PATH=path/to/your/images
   ```

   Replace `path/to/your/images` with the path to the folder containing the images you want to search.


5. Download yolo4 configuration, weights and class names.

- [yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights)


- [yolov4.cfg](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg?raw=true)


- [coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true)

## Usage
   
1. Open scout.ipynb with jupyter notebook.
    
2. Enter your search query when prompted (one word only).

3. View the search results, which will display matching images with bounding boxes around detected objects.

## Code Overview

### Classes

   

- **FaceRecognition**:  Manages face recognition tasks, including loading a pre-trained model and recognizing faces in images.
  
  #### Methods
  - `extract_face_features(image_path)`: Extracts face features from an image. This method uses the `face_recognition` library to load an image and extract face encodings. It returns a list of face encodings for any detected faces. If an error occurs during the extraction process, it logs the error and returns `None`.

- **ImageRecognition**: Handles loading the YOLOv4 model, preprocessing images, and extracting object features.
  
  #### Methods
  - `__init__(model_weights, config_file, classes_file)`: Initializes the model, loading weights and configuration files, and setting up class names.
  - `load_and_preprocess_image(image_path)`: Loads an image from the specified path and preprocesses it for YOLO detection.
  - `extract_objects(blob, confidence_threshold, image_width, image_height)`: Processes the input blob to detect objects, returning bounding boxes, class IDs, and confidence scores.

- **ImageSearchEngine**: Manages image searching based on user queries, caching features, and visualizing results.
  
  #### Methods
  - `__init__(folder_path, general_model, thres, class_file)`: Initializes the search engine, loading image features from the specified folder and cache.
  - `_load_or_extract_objects()`: Loads cached features from a file if available; otherwise, extracts features from new images and caches them.
  - `detect_objects(img)`: Extracts object features for a single image and returns the image along with its detected features.
  - `plot_image_with_boxes(filename, features)`: Visualizes a single image with bounding boxes drawn around detected objects.
  - `search_images_by_query(query)`: Searches for images containing the queried object class and returns a list of matching images.

- **FaceSearchEngine**: Class for searching images containing specific faces.

  #### Methods
  - `__init__(folder_path)`: Initializes the search engine with the specified folder path. It loads cached face features or extracts them from images in the folder.
  - `_load_or_extract_face_features()`: Loads cached face features from a file if available. If not, it extracts face features from images in the specified folder and caches them for future use.
  - `search_images_by_face(input_image_path)`: Searches for images containing faces from the input image. It compares face encodings from the input image with cached encodings and returns a list of matching image filenames. If no faces are found in the input image, it returns an empty list.


## Caching

Features are cached in a file named `image_features_cache.pkl`. If the cache file exists, it loads the features from the cache; otherwise, it extracts features from the images and caches them for future use.


