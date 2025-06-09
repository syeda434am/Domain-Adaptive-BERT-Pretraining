import os
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast
from transformers.utils import logging
from com.mhire.data_processing.pre_training_data_handler import PreTrainingDataHandler
from com.mhire.pre_training.pre_training import Pretraining

logger = logging.get_logger("transformers.trainer")
logger.setLevel(logging.INFO)

def run_pretraining():
    # Set up directories
    LOCAL_DIR = "/tmp/datasets/"
    OUTPUT_DIR = "/tmp/trained_model/"
    LOG_DIR = "/tmp/logs/"
    
    CLEAN_DATA_FILE = os.path.join(LOCAL_DIR, "mlm_format.jsonl")
    NSP_FORMAT_FILE = os.path.join(LOCAL_DIR, "nsp_format.jsonl")

    # Initialize tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", model_max_length=512)

    # Initialize DataHandler and prepare datasets
    data_handler = PreTrainingDataHandler(tokenizer)
    nsp_dataset = data_handler.prepare_nsp_dataset(file_path=NSP_FORMAT_FILE)
    mlm_dataset = data_handler.prepare_mlm_dataset(file_path=CLEAN_DATA_FILE)
    combined_dataset = data_handler.combine_datasets(mlm_dataset, nsp_dataset)

    # Split datasets into train and validation
    train_indices, val_indices = train_test_split(
        range(len(combined_dataset)), test_size=0.2, random_state=42
    )
    train_dataset = torch.utils.data.Subset(combined_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(combined_dataset, val_indices)

    # Initialize pretraining and train model
    pretrainer = Pretraining("bert-base-uncased", OUTPUT_DIR, LOG_DIR, tokenizer)
    pretrainer.train(train_dataset, val_dataset)

if __name__ == "__main__":
    run_pretraining()