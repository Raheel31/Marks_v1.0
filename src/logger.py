import logging
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_FILE = os.path.join(BASE_DIR, "..", "logs","app.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),   # Writes logs to logs/app.log
        logging.StreamHandler()          # Also prints logs to console
    ]
)

# Function to get logger for each module
def get_logger(name: str):
    """
    Logger functoin to capture all data runs

    Args:
        name (str): Receives the log frome ach function
    """
    return logging.getLogger(name)
