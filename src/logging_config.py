"""Enhanced logging configuration with rotation and structured output."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, "extra_data"):
            log_data["extra"] = record.extra_data
        
        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[35m",   # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        """Format with colors for console."""
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    log_dir: Optional[Path] = None,
    level: str = "INFO",
    console: bool = True,
    file_logging: bool = True,
    rotation: str = "size",  # 'size' or 'time'
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    structured: bool = False,
    colored: bool = True,
) -> logging.Logger:
    """Setup comprehensive logging configuration.
    
    Args:
        log_dir: Directory for log files
        level: Logging level
        console: Enable console logging
        file_logging: Enable file logging
        rotation: Rotation type ('size' or 'time')
        max_bytes: Max file size for rotation (if rotation='size')
        backup_count: Number of backup files to keep
        structured: Use structured JSON logging for files
        colored: Use colored output for console
    
    Returns:
        Configured root logger
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        if colored and sys.stdout.isatty():
            console_fmt = ColoredFormatter(
                "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        else:
            console_fmt = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        
        console_handler.setFormatter(console_fmt)
        root_logger.addHandler(console_handler)
    
    # File handlers
    if file_logging and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Main log file
        main_log = log_dir / "pipeline.log"
        
        if rotation == "size":
            file_handler = RotatingFileHandler(
                main_log,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
        else:  # time-based
            file_handler = TimedRotatingFileHandler(
                main_log,
                when="midnight",
                interval=1,
                backupCount=backup_count,
                encoding="utf-8",
            )
        
        file_handler.setLevel(logging.DEBUG)
        
        if structured:
            file_fmt = StructuredFormatter()
        else:
            file_fmt = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(name)s - [%(filename)s:%(lineno)d] - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        
        file_handler.setFormatter(file_fmt)
        root_logger.addHandler(file_handler)
        
        # Error log file
        error_log = log_dir / "errors.log"
        error_handler = RotatingFileHandler(
            error_log,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_fmt)
        root_logger.addHandler(error_handler)
    
    return root_logger


class LoggerContext:
    """Context manager for temporary logging configuration."""
    
    def __init__(self, extra_data: Optional[Dict[str, Any]] = None):
        self.extra_data = extra_data or {}
        self.original_factory = None
    
    def __enter__(self):
        self.original_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = self.original_factory(*args, **kwargs)
            record.extra_data = self.extra_data
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_factory:
            logging.setLogRecordFactory(self.original_factory)


def log_execution_time(logger: logging.Logger):
    """Decorator to log function execution time."""
    import functools
    import time
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.info(
                    "Function %s completed in %.2f seconds",
                    func.__name__,
                    elapsed,
                )
                return result
            except Exception as exc:
                elapsed = time.perf_counter() - start
                logger.error(
                    "Function %s failed after %.2f seconds: %s",
                    func.__name__,
                    elapsed,
                    exc,
                )
                raise
        
        return wrapper
    
    return decorator
