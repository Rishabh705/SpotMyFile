from src.logging import logger
from src.config import Config
from typing import Any, List, Optional, Dict
from pathlib import Path
from collections import OrderedDict
import pickle
import time
import os

os.makedirs(Config.CACHE_DIR, exist_ok=True)

class CacheManager:
    """LRU-based cache manager"""

    def __init__(self, cache_path, auto_save_interval=Config.SAVE_INTERVAL, max_cache_size=100000):
        """Initialize the cache manager"""
        self.cache_path = cache_path
        self.cache = self._load_cache()
        self.auto_save_interval = auto_save_interval
        self.last_save_time = 0
        self.items_since_save = 0
        self.max_cache_size = max_cache_size

    def _load_cache(self) -> "OrderedDict[str, Any]":
        """Load cache from disk"""
        if Path(self.cache_path).exists():
            try:
                with open(self.cache_path, 'rb') as f:
                    data = pickle.load(f)
                    if not isinstance(data, OrderedDict):
                        raise ValueError("Cache file is not an OrderedDict")
                    logger.info(f"Loaded {len(data)} items from cache at {self.cache_path}")
                    return data
            except Exception as e:
                logger.error(f"Error loading cache from {self.cache_path}: {e}. Starting with empty cache.")
                return OrderedDict()
        else:
            logger.info(f"Cache file {self.cache_path} not found. Starting with empty cache.")
            return OrderedDict()

    def save_cache(self, force=False) -> bool:
        """Save cache to disk with atomic file operations"""
        if self.items_since_save == 0 and not force:
            return True

        temp_file = f"{self.cache_path}.temp"
        try:
            start_time = time.time()
            with open(temp_file, 'wb') as f:
                pickle.dump(self.cache, f)
            os.replace(temp_file, self.cache_path)
            elapsed = time.time() - start_time
            logger.info(f"Saved {len(self.cache)} items to cache in {elapsed:.2f}s")
            self.items_since_save = 0
            self.last_save_time = time.time()
            return True
        except Exception as e:
            logger.error(f"Error saving to {self.cache_path}: {e}")
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
            return False

    def add_item(self, key: str, value: Any) -> None:
        """Add or update an item in the cache (with LRU eviction)"""
        self._add_item_internal(key, value)

    def _add_item_internal(self, key: str, value: Any) -> None:
        """Internal: Add item with eviction logic"""
        if key in self.cache:
            self.cache.move_to_end(key)  # Update LRU order
        self.cache[key] = value
        self.items_since_save += 1

        if len(self.cache) > self.max_cache_size:
            evicted_key, _ = self.cache.popitem(last=False)  # Remove LRU
            logger.debug(f"Evicted LRU item: {evicted_key}")

        if (self.items_since_save >= self.auto_save_interval and 
            time.time() - self.last_save_time > 300):
            self.save_cache()

    def get_item(self, key: str) -> Optional[Any]:
        """Retrieve item and mark as recently used"""
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def contains(self, key: str) -> bool:
        return key in self.cache

    def clear(self) -> None:
        self.cache.clear()
        self.items_since_save = 0

    def get_all_keys(self) -> List[str]:
        return list(self.cache.keys())

    def get_size(self) -> int:
        return len(self.cache)