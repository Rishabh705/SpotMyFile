import cv2
import face_recognition
import numpy as np
from src.config import Config
from src.logging import logger
from pathlib import Path
import concurrent.futures
import fitz  # PyMuPDF
import docx
import pytesseract
from PIL import Image
from tqdm import tqdm
import os

class OCRExtractor:
    """Handles OCR-based text extraction from various image formats"""
    
    @staticmethod
    def extract_text_from_image(image_path, dpi=Config.OCR_DPI, 
                               tesseract_config=Config.TESSERACT_CONFIG):
        """
        Extract text from an image file using OCR
        """
        try:
            # Open image with PIL
            img = Image.open(image_path)
            
            # Convert to grayscale to improve OCR and reduce memory usage
            img = img.convert("L")
            
            # Extract text using pytesseract
            text = pytesseract.image_to_string(img, config=tesseract_config)
            
            logger.debug(f"Extracted {len(text)} characters via OCR from {image_path}")
            return text[:Config.MAX_CHARS]
        except Exception as e:
            logger.error(f"Error extracting text via OCR from {image_path}: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_pixmap(pixmap, tesseract_config=Config.TESSERACT_CONFIG):
        """
        Extract text from a PyMuPDF pixmap using OCR
        """
        try:
            # Convert pixmap to PIL Image
            img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            
            # Convert to grayscale to improve OCR and reduce memory usage
            img = img.convert("L")
            
            # Extract text using pytesseract
            text = pytesseract.image_to_string(img, config=tesseract_config)
            
            return text
        except Exception as e:
            logger.error(f"Error extracting text via OCR from pixmap: {e}")
            return ""


class TextContentExtractor:
    """Extracts text content from various file types"""
    
    def __init__(self):
        self.ocr_extractor = OCRExtractor()
    
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
    
    def process_page(self, file_path, page_num, ocr_dpi, tesseract_config):
        """Process a single PDF page with text extraction and optional OCR"""
        try:
            with fitz.open(file_path) as doc:
                page = doc.load_page(page_num)
                text = page.get_text("text")
                if not text.strip():  # If no text found, try OCR
                    pix = page.get_pixmap(dpi=ocr_dpi)
                    text = self.ocr_extractor.extract_text_from_pixmap(pix, tesseract_config)
                return (page_num, text)
        except Exception as e:
            return (page_num, f"[ERROR on page {page_num}]: {str(e)}")
    
    def extract_text_from_pdf(self, file_path: str, ocr_dpi: int = Config.OCR_DPI, 
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
                            executor.submit(self.process_page, file_path, i, ocr_dpi, tesseract_config)
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
    
    def extract_text_from_image(self, file_path):
        """
        Extract text from common image formats (png, jpg, jpeg, tiff, bmp)
        """
        return self.ocr_extractor.extract_text_from_image(file_path)
    
    def extract_text(self, file_path):
        try:
            file_extension = Path(file_path).suffix.lower()
            logger.debug(f"Attempting to extract text from {file_path} with extension {file_extension}")
            
            result = ""
            if file_extension == ".txt":
                result = self.extract_text_from_txt(file_path)
            elif file_extension == ".pdf":
                result = self.extract_text_from_pdf(file_path)
            elif file_extension == ".docx":
                result = self.extract_text_from_docx(file_path)
            elif file_extension in Config.IMAGE_EXTENSIONS:
                result = self.extract_text_from_image(file_path)
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


class ImageFeatureExtractor:
    """Handles feature extraction for images including faces and objects"""
    
    def __init__(self):
        # Use Config values instead of hardcoded paths
        self._face_model = None
        self._object_net = None
        self._output_layers = None
        self._classes = None
        
        # Store paths from Config for lazy loading
        self.yolo_weights = Config.YOLO_WEIGHTS
        self.yolo_config = Config.YOLO_CONFIG
        self.yolo_classes = Config.YOLO_CLASSES
        self.confidence_threshold = Config.YOLO_CONFIDENCE
        self.face_similarity_threshold = Config.FACE_SIMILARITY_THRESHOLD

    @property
    def face_model(self):
        """Lazy loading of face recognition model"""
        if self._face_model is None:
            import face_recognition
            self._face_model = face_recognition
        return self._face_model
    
    @property
    def object_net(self):
        """Lazy loading of object detection model"""
        if self._object_net is None:
            self._object_net = cv2.dnn.readNet(self.yolo_weights, self.yolo_config)
            layer_names = self._object_net.getLayerNames()
            unconnected_layers = self._object_net.getUnconnectedOutLayers().flatten()
            self._output_layers = [layer_names[i-1] for i in unconnected_layers]
        return self._object_net
    
    @property
    def output_layers(self):
        """Get output layers, initializing if needed"""
        if self._output_layers is None:
            _ = self.object_net  
        return self._output_layers
    
    @property
    def classes(self):
        """Lazy loading of class names"""
        if self._classes is None:
            with open(self.yolo_classes, "r") as f:
                self._classes = [line.strip() for line in f.readlines()]
        return self._classes

    def extract_face_features(self, image_path):
        """Extract face encodings from an image"""
        try:
            image = self.face_model.load_image_file(image_path)
            return self.face_model.face_encodings(image)
        except Exception as e:
            logger.error(f"Face extraction error for {image_path}: {e}")
            return []

    def extract_object_features(self, image_path):
        """Detect objects using YOLOv4 and return features"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Could not read image: {image_path}")
                return None
                
            height, width, _ = img.shape
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), 
                                    (0, 0, 0), True, crop=False)
            
            self.object_net.setInput(blob)
            outputs = self.object_net.forward(self.output_layers)
            
            # Process outputs
            boxes = []
            confidences = []
            class_ids = []
            
            for out in outputs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    # Use confidence threshold from Config
                    if confidence > self.confidence_threshold:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            return {
                'boxes': boxes,
                'confidences': confidences,
                'class_ids': class_ids,
            }
            
        except Exception as e:
            logger.error(f"Object detection error for {image_path}: {e}")
            return None