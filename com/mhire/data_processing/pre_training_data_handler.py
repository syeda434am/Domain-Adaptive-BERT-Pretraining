import torch
from datasets import load_dataset
from transformers import TextDatasetForNextSentencePrediction
from torch.utils.data import Dataset
import logging, json

class PreTrainingDataHandler:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def prepare_nsp_dataset(self, file_path, block_size=512):
        """
        Use TextDatasetForNextSentencePrediction for NSP data preparation.
        """
        try:
            logging.info(f"Preparing NSP dataset from {file_path}")
            return TextDatasetForNextSentencePrediction(
                tokenizer=self.tokenizer,
                file_path=file_path,
                block_size=block_size,
                overwrite_cache=True,
            )
        except Exception as e:
            logging.error(f"Error preparing NSP dataset: {str(e)}")
            raise

    def prepare_mlm_dataset(self, file_path, max_length=512):
        """
        Prepare MLM dataset using Hugging Face's load_dataset utility.
        """
        try:
            logging.info(f"Preparing MLM dataset from {file_path}")
            mlm_data = load_dataset("json", data_files=file_path, split="train")
            mlm_data = mlm_data.map(
                lambda x: self.tokenizer(
                    x["sentence"],
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                ),
                batched=True,
            )
            mlm_data = mlm_data.map(lambda x: {"labels": x["input_ids"]}, batched=True)
            mlm_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
            return mlm_data
        except Exception as e:
            logging.error(f"Error preparing MLM dataset: {str(e)}")
            raise

    def combine_datasets(self, mlm_dataset, nsp_dataset):
        class CombinedDataset(Dataset):
            def __init__(self, mlm_dataset, nsp_dataset):
                self.mlm_dataset = mlm_dataset
                self.nsp_dataset = nsp_dataset

            def __len__(self):
                return max(len(self.mlm_dataset), len(self.nsp_dataset))

            def __getitem__(self, idx):
                mlm_example = self.mlm_dataset[idx % len(self.mlm_dataset)]
                nsp_example = self.nsp_dataset[idx % len(self.nsp_dataset)]

                input_ids = mlm_example["input_ids"]
                attention_mask = mlm_example["attention_mask"]
                labels = mlm_example["labels"]

                token_type_ids = nsp_example.get("token_type_ids", torch.zeros_like(input_ids))
                token_type_ids = token_type_ids[:len(input_ids)]
                token_type_ids = torch.nn.functional.pad(token_type_ids, (0, len(input_ids) - len(token_type_ids)))

                next_sentence_label = nsp_example.get("next_sentence_label", -1)

                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                    "labels": labels,
                    "next_sentence_label": next_sentence_label,
                }

        logging.info("Combining MLM and NSP datasets")
        return CombinedDataset(mlm_dataset, nsp_dataset)
