import os
import logging
import pandas as pd
from datasets import load_from_disk, Dataset, concatenate_datasets

# Reuse deduplication function from pipelines.py
from pipelines import deduplicate_df  

logger = logging.getLogger("dataset_merger")
logging.basicConfig(level=logging.INFO)

DATA_DIR = "preprocess_outputs/en-am/include"


def merge_and_deduplicate(data_dir=DATA_DIR, src_col="en", tgt_col="am") -> Dataset:
    """
    Merge all HF datasets under data_dir and deduplicate using deduplicate_df.
    """
    logger.info(f"Looking for datasets in: {data_dir}")

    datasets = []
    for folder in os.listdir(data_dir):
        path = os.path.join(data_dir, folder)
        if os.path.isdir(path):
            try:
                ds = load_from_disk(path)
                logger.info(f"Loaded dataset from {path} → {len(ds)} rows")
                datasets.append(ds)
            except Exception as e:
                logger.warning(f"Skipping {path}: {e}")

    if not datasets:
        raise ValueError(f"No datasets found in {data_dir}")

    # Merge all datasets
    merged = concatenate_datasets(datasets)
    logger.info(f"Merged dataset size: {len(merged)} rows")

    # Convert to pandas for deduplication
    df = merged.to_pandas()[[src_col, tgt_col]]
    df = df.rename(columns={src_col: "Source", tgt_col: "Target"})
    df = deduplicate_df(df)
    df = df.rename(columns={"Source": src_col, "Target": tgt_col})

    # Convert back to HF dataset
    deduped = Dataset.from_pandas(df, preserve_index=False)
    logger.info(f"Deduplicated dataset size: {len(deduped)} rows")

    return deduped


if __name__ == "__main__":
    merged_dataset = merge_and_deduplicate()
    merged_dataset.save_to_disk("preprocess_outputs/en-am-merged")
    logger.info("Saved merged and deduplicated dataset to preprocess_outputs/en-am-merged")