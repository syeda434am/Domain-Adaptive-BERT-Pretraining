import os
import json
from logging import info as log
from PyPDF2 import PdfReader

class PDFParser:
    def __init__(self):
        self.reader = None

    def parse_pdfs(self, input_dir, output_dir):
        """Parses PDFs in the input directory and creates JSONL file with line by line text."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for pdf_file in os.listdir(input_dir):
            if pdf_file.endswith(".pdf"):
                input_path = os.path.join(input_dir, pdf_file)
                output_path = os.path.join(output_dir, pdf_file.replace(".pdf", ".jsonl"))

                with open(input_path, 'rb') as f:
                    self.reader = PdfReader(f)
                    text_lines = []
                    for page in self.reader.pages:
                        text_lines.extend(page.extract_text().splitlines())

                with open(output_path, 'w', encoding='utf-8') as outfile:
                    for line in text_lines:
                        json.dump({'sentence': line.strip()}, outfile)
                        outfile.write('\n')
                log(f"Processed PDF and saved JSONL: {output_path}")
