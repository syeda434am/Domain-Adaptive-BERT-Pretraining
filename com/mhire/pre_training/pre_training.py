import torch
from transformers import BertForPreTraining, DataCollatorForLanguageModeling, Trainer, TrainingArguments

class Pretraining:
    def __init__(self, model_name, output_dir, log_dir, tokenizer):
        self.tokenizer = tokenizer
        self.model = self.initialize_model(model_name)
        self.output_dir = output_dir
        self.log_dir = log_dir

    def initialize_model(self, model_name):
        return BertForPreTraining.from_pretrained(model_name)

    def create_data_collator(self, mlm_probability=0.15):
        mlm_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=mlm_probability
        )
        
        def collate_fn(batch):
            mlm_inputs = [{
                "input_ids": item["input_ids"],
                "attention_mask": item["attention_mask"]
            } for item in batch]

            mlm_outputs = mlm_collator(mlm_inputs)
            next_sentence_labels = torch.tensor([item["next_sentence_label"] for item in batch])
            mlm_outputs["next_sentence_label"] = next_sentence_labels
            return mlm_outputs
        
        return collate_fn

    def create_training_args(self, epochs=3, batch_size=8):
        return TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_steps=1000,
            save_total_limit=2,
            logging_dir=self.log_dir,
            logging_strategy="steps",
            logging_steps=5,
            report_to="tensorboard",
            eval_strategy="steps",
            eval_steps=10,
            save_strategy="steps"
        )

    def train(self, train_dataset, val_dataset, mlm_probability=0.15, epochs=3, batch_size=8):
        data_collator = self.create_data_collator(mlm_probability)
        training_args = self.create_training_args(epochs, batch_size)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator
        )

        trainer.train()
        self.save_model()

    def save_model(self):
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"Model saved to {self.output_dir}")