# Domain-Adaptive BERT Pretraining

This repository contains the code for pretraining a BERT model on domain-specific data.

## Project Structure

- `com/mhire/data_processing/`: Contains scripts for data preparation, including PDF parsing, sentence chunking, and Next Sentence Prediction (NSP) data generation.
- `com/mhire/pre_training/`: Contains the core pretraining implementation using the `transformers` library.
- `com/mhire/pdf_processing_pipeline.py`: Orchestrates the PDF processing steps to prepare data for pretraining.
- `com/mhire/pre_training_runner.py`: The main script to initiate the pretraining process.
- `com/mhire/utility/`: Contains utility functions for directory management, Google Cloud Storage interactions, NLTK data handling, and zip operations.

## Pretraining Overview

The pretraining pipeline involves:
1. **PDF Processing**: Parsing PDF documents to extract text and prepare it for model input.
2. **Data Preparation**: Generating Masked Language Model (MLM) and Next Sentence Prediction (NSP) training examples.
3. **Model Pretraining**: Training a BERT model using the prepared data.

After domain-adaptive pretraining, the validation result for the MLM/NSP task was 89% accurate.

## 📦 Dataset

You can access the pretraining dataset on Kaggle:  
➡️ [Medical Domain Corpus](https://www.kaggle.com/datasets/aunanya875/medical-domain-corpus)

The dataset contains cleaned and parsed sentences from globally recognized medical textbooks and peer-reviewed journals.

## How to Run

### Prerequisites

- Python 3.x
- pip (Python package installer)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/syeda434am/Domain-Adaptive-BERT-Pretraining.git
   cd Domain-Adaptive-BERT-Pretraining
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### Data Preparation

Ensure your PDF documents are placed in the designated input directory (as configured in `pdf_processing_pipeline.py`). The pipeline will process these PDFs and generate the necessary JSONL files for pretraining.

To run the PDF processing pipeline:
```bash
python com/mhire/pdf_processing_pipeline.py
```

### Running Pretraining

Once the data is prepared, you can start the pretraining process. The `pre_training_runner.py` script handles the entire pretraining workflow.

To run the pretraining:
```bash
python com/mhire/pre_training_runner.py
```

### Configuration

Configuration parameters for pretraining (e.g., model name, batch size, epochs, output directories) can be adjusted within `pre_training_runner.py` and `pre_training/pre_training.py`.

## Results

Upon completion of the domain-adaptive pretraining, the model achieved an 89% accuracy on the validation set for the Masked Language Model (MLM) and Next Sentence Prediction (NSP) tasks.

## Citing This Repository

If you use this repository for your research or projects, please consider citing it. This project is licensed under the Apache 2.0 License.

```bibtex
@misc{domain_adaptive_bert_pretraining,
  title={Domain-Adaptive BERT Pretraining},
  author={Syeda Aunanya Mahmud},
  year={2025},
  publisher={GitHub},
  url={https://github.com/syeda434am/Domain-Adaptive-BERT-Pretraining},
  license={Apache 2.0},
}
```