from data_processing.data_preparation import DataPreparation
from logging import info as log
import json
import random
import os, regex as re
class NSPGenerator:
    """Handles Next Sentence Prediction (NSP) pair generation."""
    @staticmethod
    def generate_nsp_pairs(sentences, output_file, start_index=0):
        """Generates positive and negative sentence pairs for NSP."""
        nsp_data = []
        # Create positive pairs (next sentence is correct)
        for i in range(start_index, len(sentences) - 1):
            nsp_data.append({
                "sentence_a": sentences[i],
                "sentence_b": sentences[i + 1],
                "label": 1
            })
        # Create negative pairs (next sentence is random)
        for i in range(start_index, len(sentences) - 1):
            random_index = random.randint(0, len(sentences) - 1)
            while random_index == i or random_index == i + 1:
                random_index = random.randint(0, len(sentences) - 1)
            nsp_data.append({
                "sentence_a": sentences[i],
                "sentence_b": sentences[random_index],
                "label": 0
            })
        # Shuffle NSP pairs for randomness
        random.shuffle(nsp_data)
        # Write the NSP data to a file
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for record in nsp_data:
                record['sentence_a'] = DataPreparation.clean_sentence(record['sentence_a'])
                record['sentence_b'] = DataPreparation.clean_sentence(record['sentence_b'])              
                outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
        log(f"NSP dataset created and saved to {output_file}")
    @staticmethod
    def generate_nsp_from_directory(input_dir, output_dir, nsp_output_file, max_tokens=512):
        """Processes all JSONL files in the input directory, splits sentences into chunks, and generates NSP pairs."""
        sentences = []
        # Process each file in the input directory
        for jsonl_file in os.listdir(input_dir):
            if jsonl_file.endswith(".jsonl"):
                input_path = os.path.join(input_dir, jsonl_file)
                output_path = os.path.join(output_dir, jsonl_file)
                with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
                    for line in infile:
                        data = json.loads(line.strip())
                        text = data.get('text', '')
                        if text:
                            # Split the text into sentences, then into chunks
                            sentences_in_file = [sentence.strip() for sentence in re.split(r'[.!?]', text) if sentence]
                            for sentence in sentences_in_file:
                                chunks = DataPreparation.split_sentence_into_chunks(sentence, max_tokens)
                                for chunk in chunks:
                                    sentences.append(chunk)
                                    outfile.write(json.dumps({'sentence': chunk}) + '\n')
                log(f"Processed file: {jsonl_file}")
        # After processing files, generate NSP pairs
        NSPGenerator.generate_nsp_pairs(sentences, nsp_output_file)