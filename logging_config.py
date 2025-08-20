"""
Logging configuration for the Context-Aware Research Chatbot
"""
import os
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import Optional
import json

from config import config

def setup_logging(
    log_level: str = None,
    log_file: str = None,
    enable_file_logging: bool = True,
    enable_console_logging: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup comprehensive logging for the chatbot system
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (default: logs/chatbot.log)
        enable_file_logging: Whether to log to file
        enable_console_logging: Whether to log to console
        max_file_size: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
    
    Returns:
        Configured logger instance
    """
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Set log level
    if log_level is None:
        log_level = config.log_level
    
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger("chatbot")
    logger.setLevel(numeric_level)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(module)s:%(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    if enable_console_logging:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if enable_file_logging:
        if log_file is None:
            log_file = logs_dir / "chatbot.log"
        
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    # Error file handler (only errors and above)
    error_file = logs_dir / "errors.log"
    error_handler = logging.handlers.RotatingFileHandler(
        filename=error_file,
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    logger.addHandler(error_handler)
    
    logger.info(f"Logging initialized with level: {log_level}")
    return logger

class StructuredLogger:
    """Structured logging for better analysis and monitoring"""
    
    def __init__(self, logger_name: str = "chatbot.structured"):
        self.logger = logging.getLogger(logger_name)
        self.setup_json_logging()
    
    def setup_json_logging(self):
        """Setup JSON structured logging"""
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # JSON log file
        json_file = logs_dir / "structured.json"
        json_handler = logging.handlers.RotatingFileHandler(
            filename=json_file,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10,
            encoding='utf-8'
        )
        json_handler.setLevel(logging.INFO)
        
        # Custom JSON formatter
        json_formatter = JsonFormatter()
        json_handler.setFormatter(json_formatter)
        
        self.logger.addHandler(json_handler)
        self.logger.setLevel(logging.INFO)
    
    def log_chat_interaction(self, session_id: str, user_message: str, 
                           bot_response: str, tool_used: str, 
                           response_time: float, sources_count: int):
        """Log chat interaction with structured data"""
        self.logger.info("chat_interaction", extra={
            "event_type": "chat_interaction",
            "session_id": session_id,
            "user_message_length": len(user_message),
            "bot_response_length": len(bot_response),
            "tool_used": tool_used,
            "response_time_seconds": response_time,
            "sources_count": sources_count,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def log_system_event(self, event_type: str, details: dict):
        """Log system events"""
        self.logger.info(event_type, extra={
            "event_type": event_type,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def log_error(self, error_type: str, error_message: str, 
                  context: dict = None):
        """Log errors with context"""
        self.logger.error("system_error", extra={
            "event_type": "system_error",
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {},
            "timestamp": datetime.utcnow().isoformat()
        })

class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record):
        """Format log record as JSON"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if they exist
        if hasattr(record, 'event_type'):
            log_entry["event_type"] = record.event_type
        
        # Add any extra fields from the log record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 
                          'exc_text', 'stack_info', 'message']:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)

class PerformanceLogger:
    """Logger for performance metrics"""
    
    def __init__(self):
        self.logger = logging.getLogger("chatbot.performance")
        self.setup_performance_logging()
    
    def setup_performance_logging(self):
        """Setup performance logging"""
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Performance log file
        perf_file = logs_dir / "performance.log"
        perf_handler = logging.handlers.RotatingFileHandler(
            filename=perf_file,
            maxBytes=20 * 1024 * 1024,  # 20MB
            backupCount=5,
            encoding='utf-8'
        )
        perf_handler.setLevel(logging.INFO)
        
        # Performance formatter
        perf_formatter = logging.Formatter(
            '%(asctime)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        perf_handler.setFormatter(perf_formatter)
        
        self.logger.addHandler(perf_handler)
        self.logger.setLevel(logging.INFO)
    
    def log_response_time(self, endpoint: str, method: str, 
                         response_time: float, status_code: int):
        """Log API response times"""
        self.logger.info(
            f"API | {method} {endpoint} | {response_time:.3f}s | {status_code}"
        )
    
    def log_tool_performance(self, tool_name: str, query_length: int, 
                           processing_time: float, success: bool):
        """Log tool performance metrics"""
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(
            f"TOOL | {tool_name} | {query_length}chars | {processing_time:.3f}s | {status}"
        )
    
    def log_database_operation(self, operation: str, table: str, 
                             execution_time: float, record_count: int = None):
        """Log database operation performance"""
        records = f" | {record_count}records" if record_count is not None else ""
        self.logger.info(
            f"DB | {operation} {table} | {execution_time:.3f}s{records}"
        )

def log_system_startup():
    """Log system startup information"""
    logger = logging.getLogger("chatbot.startup")
    structured_logger = StructuredLogger()
    
    startup_info = {
        "python_version": os.sys.version,
        "working_directory": str(Path.cwd()),
        "config": {
            "llm_model": config.llm_model,
            "vector_store_type": config.vector_store_type,
            "api_port": config.api_port,
            "log_level": config.log_level
        }
    }
    
    logger.info("System startup initiated")
    structured_logger.log_system_event("system_startup", startup_info)

def log_system_shutdown():
    """Log system shutdown"""
    logger = logging.getLogger("chatbot.shutdown")
    structured_logger = StructuredLogger()
    
    logger.info("System shutdown initiated")
    structured_logger.log_system_event("system_shutdown", {
        "shutdown_time": datetime.utcnow().isoformat()
    })

# Context managers for performance logging
class LogExecutionTime:
    """Context manager to log execution time"""
    
    def __init__(self, operation_name: str, logger: logging.Logger = None):
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger("chatbot.performance")
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.utcnow()
        duration = (end_time - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.info(f"{self.operation_name} completed in {duration:.3f}s")
        else:
            self.logger.error(f"{self.operation_name} failed after {duration:.3f}s: {exc_val}")

# Initialize loggers
def initialize_logging():
    """Initialize all logging components"""
    # Main application logger
    main_logger = setup_logging()
    
    # Structured logger for analytics
    structured_logger = StructuredLogger()
    
    # Performance logger
    performance_logger = PerformanceLogger()
    
    # Log startup
    log_system_startup()
    
    return main_logger, structured_logger, performance_logger

# Export commonly used loggers
logger = logging.getLogger("chatbot")
structured_logger = StructuredLogger()
performance_logger = PerformanceLogger()

if __name__ == "__main__":
    # Test logging setup
    test_logger = setup_logging("DEBUG")
    test_logger.debug("Debug message")
    test_logger.info("Info message")
    test_logger.warning("Warning message")
    test_logger.error("Error message")
    
    # Test structured logging
    struct_logger = StructuredLogger()
    struct_logger.log_chat_interaction(
        session_id="test-123",
        user_message="Test question",
        bot_response="Test response",
        tool_used="rag",
        response_time=1.5,
        sources_count=3
    )
    
    # Test performance logging
    perf_logger = PerformanceLogger()
    perf_logger.log_response_time("/chat", "POST", 2.3, 200)
    
    print("Logging test completed. Check logs/ directory for output files.")