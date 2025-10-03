"""Caching module for intermediate results."""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class CacheManager:
    """Manager for caching intermediate pipeline results."""
    
    def __init__(self, cache_dir: Path = Path(".cache"), enabled: bool = True):
        """Initialize cache manager.
        
        Args:
            cache_dir: Directory for cache files
            enabled: Whether caching is enabled
        """
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cache initialized at: %s", self.cache_dir)
    
    def _get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        # Create deterministic string from args and kwargs
        key_data = {
            "args": [str(arg) for arg in args],
            "kwargs": {k: str(v) for k, v in sorted(kwargs.items())},
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str, file_type: str = "pickle") -> Optional[Any]:
        """Retrieve cached value.
        
        Args:
            key: Cache key
            file_type: File type ('pickle', 'csv', 'json')
        
        Returns:
            Cached value or None if not found
        """
        if not self.enabled:
            return None
        
        cache_file = self.cache_dir / f"{key}.{file_type}"
        
        if not cache_file.exists():
            return None
        
        try:
            if file_type == "pickle":
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            elif file_type == "csv":
                return pd.read_csv(cache_file)
            elif file_type == "json":
                with open(cache_file, "r") as f:
                    return json.load(f)
            else:
                logger.warning("Unsupported cache file type: %s", file_type)
                return None
        except Exception as exc:
            logger.warning("Failed to load cache %s: %s", key, exc)
            return None
    
    def set(self, key: str, value: Any, file_type: str = "pickle") -> bool:
        """Store value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            file_type: File type ('pickle', 'csv', 'json')
        
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False
        
        cache_file = self.cache_dir / f"{key}.{file_type}"
        
        try:
            if file_type == "pickle":
                with open(cache_file, "wb") as f:
                    pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
            elif file_type == "csv":
                if isinstance(value, pd.DataFrame):
                    value.to_csv(cache_file, index=False)
                else:
                    logger.warning("CSV caching requires DataFrame")
                    return False
            elif file_type == "json":
                with open(cache_file, "w") as f:
                    json.dump(value, f, indent=2, default=str)
            else:
                logger.warning("Unsupported cache file type: %s", file_type)
                return False
            
            logger.debug("Cached result: %s", key)
            return True
        except Exception as exc:
            logger.warning("Failed to cache %s: %s", key, exc)
            return False
    
    def clear(self, pattern: Optional[str] = None):
        """Clear cache files.
        
        Args:
            pattern: Optional glob pattern to match files (e.g., "*.pickle")
        """
        if not self.enabled or not self.cache_dir.exists():
            return
        
        try:
            if pattern:
                files = list(self.cache_dir.glob(pattern))
            else:
                files = list(self.cache_dir.iterdir())
            
            for file in files:
                if file.is_file():
                    file.unlink()
            
            logger.info("Cleared %d cache files", len(files))
        except Exception as exc:
            logger.warning("Failed to clear cache: %s", exc)
    
    def invalidate(self, key: str):
        """Invalidate specific cache entry."""
        try:
            for ext in ["pickle", "csv", "json"]:
                cache_file = self.cache_dir / f"{key}.{ext}"
                if cache_file.exists():
                    cache_file.unlink()
                    logger.debug("Invalidated cache: %s", key)
        except Exception as exc:
            logger.warning("Failed to invalidate cache %s: %s", key, exc)


# Global cache instance
_global_cache: Optional[CacheManager] = None


def get_cache() -> CacheManager:
    """Get global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = CacheManager()
    return _global_cache


def cached(
    key_prefix: str = "",
    file_type: str = "pickle",
    cache_manager: Optional[CacheManager] = None,
):
    """Decorator for caching function results.
    
    Args:
        key_prefix: Prefix for cache key
        file_type: File type for caching
        cache_manager: Optional cache manager (uses global if None)
    
    Example:
        @cached(key_prefix="load_data")
        def load_data(path):
            return pd.read_csv(path)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = cache_manager or get_cache()
            
            if not cache.enabled:
                return func(*args, **kwargs)
            
            # Generate cache key
            key_suffix = cache._get_cache_key(*args, **kwargs)
            cache_key = f"{key_prefix}_{func.__name__}_{key_suffix}"
            
            # Try to get from cache
            cached_value = cache.get(cache_key, file_type)
            if cached_value is not None:
                logger.debug("Cache hit: %s", cache_key)
                return cached_value
            
            # Compute value
            logger.debug("Cache miss: %s", cache_key)
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.set(cache_key, result, file_type)
            
            return result
        
        return wrapper
    
    return decorator


def cache_dataframe(
    df: pd.DataFrame,
    name: str,
    cache_dir: Path = Path(".cache"),
) -> Path:
    """Cache DataFrame to disk.
    
    Args:
        df: DataFrame to cache
        name: Name for cache file
        cache_dir: Cache directory
    
    Returns:
        Path to cached file
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_path = cache_dir / f"{name}.parquet"
    df.to_parquet(cache_path, index=False, compression="snappy")
    
    logger.debug("Cached DataFrame to: %s", cache_path)
    return cache_path


def load_cached_dataframe(
    name: str,
    cache_dir: Path = Path(".cache"),
) -> Optional[pd.DataFrame]:
    """Load cached DataFrame.
    
    Args:
        name: Name of cached file
        cache_dir: Cache directory
    
    Returns:
        Cached DataFrame or None if not found
    """
    cache_path = Path(cache_dir) / f"{name}.parquet"
    
    if not cache_path.exists():
        return None
    
    try:
        df = pd.read_parquet(cache_path)
        logger.debug("Loaded cached DataFrame from: %s", cache_path)
        return df
    except Exception as exc:
        logger.warning("Failed to load cached DataFrame %s: %s", name, exc)
        return None
