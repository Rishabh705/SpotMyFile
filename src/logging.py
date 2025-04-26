import logging
import os
from src.config import Config

os.makedirs(Config.LOG_DIR, exist_ok=True)
class Logger:
    """Centralized logging configuration"""

    _instance = None

    @classmethod
    def get_logger(cls):
        """Get or create a logger instance"""
        if cls._instance is None:
            # Configure logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(os.path.join(Config.LOG_DIR, 'search_engine.log')),
                    logging.StreamHandler()
                ]
            )
            cls._instance = logging.getLogger()  # Root logger
        return cls._instance

# Get logger instance
logger = Logger.get_logger()