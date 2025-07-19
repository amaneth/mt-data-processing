# Multilingual Parallel Corpus Pipeline

A unified pipeline to download, preprocess, assess the quality, merge, and push multilingual parallel corpora to the Hugging Face Hub. Supports sources like Hugging Face Datasets, GitHub, and OPUS.

---

## ✨ Features

- **Download datasets from**:
  - Hugging Face Hub (`hf`)
  - GitHub (`github`)
  - OPUS (`opus`)
- **Preprocessing**:
  - Rule-based filtering
  - Semantic filtering
- **Quality Assessment**:
  - Language identification
  - Model-based quality estimation
- **Merge and Push**:
  - Combine all processed datasets into one
  - Push to Hugging Face Hub

---

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
3. **Configure settings**
    Modify the `config.yaml` file to define your language pair, data sources, and pipeline settings.
4. **Preprocess the dataset**
    ```bash
    python process.py --config am_config.yaml
5. **Quality estimation**
    ```bash
    python quality_estimation.py --dataset data --task qe
    # or
    python quality_estimation.py --dataset data --task lang

6. **Merge the datasets**
    ```bash
    python merge.py --datasets dataset1 dataset2 ...
7. **Push to Hugging Face Hub**
    ```bash
    python push_to_hub.py --dataset data


## 🛠️ `config.yaml` Overview

The `config.yaml` file controls the entire pipeline. Here’s an overview of its sections:

### `dataset`
- `lang_pair`: Source and target languages (e.g., `en-am`)
- `sources`: List of dataset types to include (`hf`, `github`, `opus`)

#### Hugging Face datasets (`hf`)
- `name`: Identifier for the dataset
- `path`: HF dataset ID
- `split`: Train/test/dev
- `config_name`: Config name if needed
- `src_col` / `tgt_col`: Source and target language fields

#### GitHub datasets (`github`)
- `name`: Identifier for the dataset
- `src_url`: URL to the source file
- `tgt_url`: URL to the target file

#### OPUS datasets (`opus`)
- `name`: Identifier for the dataset
- `url`: Download URL

---

### `preprocessing`
- `pipelines`: List of preprocessing steps, e.g., `rule_filter`, `semantic_filter`

### `filters`
- **Rule filter**:
  - `min_length`, `max_length`
  - `max_length_ratio`
- **Semantic filter**:
  - `threshold`: Similarity threshold
  - `chunk_size`: Batch size for filtering

---

### `output`
- `prefix`: Prefix for filtered dataset files
- `format`: Final dataset format (e.g., `json`, `csv`, `parquet`)
- `save_dir`: Output directory for saving results

---

### `logging`
- `log_file`: Log filename
- `log_dir`: Directory for storing logs
- `level`: Log level (`INFO`, `DEBUG`, etc.)



