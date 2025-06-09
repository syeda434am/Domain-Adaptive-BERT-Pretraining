# This module ensures that the necessary NLTK data is available for processing text.
# - `ensure_nltk_data`: Checks if the NLTK 'punkt' tokenizer is installed, and downloads it if missing.

import nltk
from logging import info as log

def ensure_nltk_data():
    """Ensure NLTK data is available."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        log("Downloaded NLTK tokenizer data.")
