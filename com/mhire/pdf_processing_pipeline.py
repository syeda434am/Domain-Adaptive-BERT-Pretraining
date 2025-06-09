from logging import basicConfig, INFO
from com.mhire.data_processing.pdf_parser import PDFParser
from com.mhire.data_processing.data_preparation import DataPreparation
from com.mhire.data_processing.nsp_formatter import NSPGenerator
from com.mhire.utility.directory_management import create_directories, cleanup_directories
from com.mhire.utility.ntlk_util import ensure_nltk_data

# Setup logging
basicConfig(level=INFO)

# Configuration
INPUT_PDF_DIR = 'tmp/input/local_pdfs'
INTERMEDIATE_JSONL_DIR = 'tmp/intermediate/local_jsonls'
INTERMEDIATE_PROCESSED_JSONL_DIR = 'tmp/intermediate/processed_jsonls'
OUTPUT_JSONL_DIR = '/tmp/datasets/'
MLM_OUTPUT_FILE = '/tmp/datasets/mlm_format.jsonl'
NSP_OUTPUT_FILE = '/tmp/datasets/nsp_format.jsonl'

DIRECTORIES = [
    'tmp',
    'tmp/input',
    'tmp/input/local_pdfs',
    'tmp/intermediate',
    'tmp/intermediate/local_jsonls',
    'tmp/intermediate/processed_jsonls',
    '/tmp/datasets/'
]

def main():
    """Main function for processing PDFs into a merged JSONL for MLM pretraining."""
    try:
        # Ensure required NLTK data
        ensure_nltk_data()

        # Create required directories
        create_directories(DIRECTORIES)

        # Step 1: Parse PDFs into JSONL format (line by line)
        pdf_parser = PDFParser()
        pdf_parser.parse_pdfs(INPUT_PDF_DIR, INTERMEDIATE_JSONL_DIR)

        # Step 2: Extract sentences from JSONL files and split them into chunks
        data_preparation = DataPreparation()
        data_preparation.process_all_files_in_directory(
            INTERMEDIATE_JSONL_DIR, 
            INTERMEDIATE_PROCESSED_JSONL_DIR, 
            max_tokens=512
        )

        # Step 3: Merge all sentence-only JSONLs into one file
        data_preparation.combine_jsonl_files(INTERMEDIATE_PROCESSED_JSONL_DIR, MLM_OUTPUT_FILE)

        # Step 4: Generate NSP dat# Step 4: Generate NSP dataset
        NSPGenerator.generate_nsp_from_directory(MLM_OUTPUT_FILE, OUTPUT_JSONL_DIR, NSP_OUTPUT_FILE)

        # Step 5: Cleanup directories after successful task completion
        cleanup_directories([INPUT_PDF_DIR, INTERMEDIATE_JSONL_DIR, INTERMEDIATE_PROCESSED_JSONL_DIR])

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
