from src.logging import logger
from src.config import Config
from typing import Any, Dict, List, Optional, Set, Tuple
from pathlib import Path

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
