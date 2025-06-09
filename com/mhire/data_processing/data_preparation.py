# This file handles the second level of parsing, where the extracted text from the first parsing is split into sentences.
# It ensures that each sentence is stored in a JSONL file and splits longer sentences into smaller chunks if they exceed the 512-token limit.

import os
import json
import re
from logging import info as log

class DataPreparation:
    @staticmethod
    def split_sentence_into_chunks(sentence, max_tokens=512):
        """Splits a sentence into chunks with a maximum number of tokens (words)."""
        words = sentence.split()
        chunks, current_chunk, current_length = [], [], 0

        for word in words:
            word_length = len(word) + 1
            if current_length + word_length <= max_tokens:
                current_chunk.append(word)
                current_length += word_length
            else:
                chunks.append(' '.join(current_chunk))
                current_chunk, current_length = [word], word_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    @staticmethod
    def process_all_files_in_directory(input_dir, output_dir, max_tokens=512):
        """Processes all files in a directory, extracting sentences and splitting long sentences."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for jsonl_file in os.listdir(input_dir):
            if jsonl_file.endswith(".jsonl"):
                input_path = os.path.join(input_dir, jsonl_file)
                output_path = os.path.join(output_dir, jsonl_file)

                with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
                    for line in infile:
                        data = json.loads(line.strip())
                        sentences = [sentence.strip() for sentence in re.split(r'[.!?]', data['text']) if sentence]
                        
                        for sentence in sentences:
                            chunks = DataPreparation.split_sentence_into_chunks(sentence, max_tokens)
                            for chunk in chunks:
                                outfile.write(json.dumps({'sentence': chunk}) + '\n')
                log(f"Processed file: {jsonl_file}")

    @staticmethod
    def combine_jsonl_files(input_dir, output_file):
        """Combines all individual JSONL files into one."""
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for jsonl_file in os.listdir(input_dir):
                if jsonl_file.endswith(".jsonl"):
                    input_path = os.path.join(input_dir, jsonl_file)
                    with open(input_path, 'r', encoding='utf-8') as infile:
                        for line in infile:
                            outfile.write(line)
                    log(f"Merged file: {jsonl_file}")
