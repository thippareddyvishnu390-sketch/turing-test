import logging
import sys
from typing import Final

# Constants for logging configuration
LOG_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"


def setup_logging() -> None:
    """
    Configures the root logger for the application.
    
    Sets the logging level to INFO and directs output to the console (stdout)
    using a standardized format.
    """
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Ensures this configuration is applied even if logging was previously configured
    )


def get_logger(name: str) -> logging.Logger:
    """
    Retrieves a logger instance for the specified name.
    
    Args:
        name: The name of the logger, typically the module's __name__.
        
    Returns:
        logging.Logger: A configured logger instance.
    """
    return logging.getLogger(name)


# Initialize logging configuration on module load
setup_logging()