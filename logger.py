import logging
from datetime import datetime
import os

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Setup logger
logger = logging.getLogger("SentimentAnalysisLogger")
logger.setLevel(logging.INFO)  # Set to INFO to capture both INFO and ERROR

# Create file handler with time-stamped log filename
log_file = f"logs/sentiment_analysis_{datetime.now().strftime('%Y-%m-%d')}.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
# logger.addHandler(console_handler)
