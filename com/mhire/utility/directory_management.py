# This file contains functions to create and clean up 
# the necessary directories for storing the input PDFs, 
# intermediate JSONL files, and output JSONL files.

import os
import shutil
from logging import info as log

def create_directories(DIRECTORIES=[]):
    """Ensure all required directories exist."""
    for directory in DIRECTORIES:
        os.makedirs(directory, exist_ok=True)
        log(f"Ensured directory exists: {directory}")

def cleanup_directories(directories):
    """Deletes specified directories and their contents."""
    for directory in directories:
        try:
            if os.path.exists(directory):
                shutil.rmtree(directory)  # Deletes the directory and all its contents
                log(f"Deleted directory: {directory}")
            else:
                log(f"Directory does not exist, skipping: {directory}")
        except Exception as e:
            log(f"Error deleting directory {directory}: {e}")
