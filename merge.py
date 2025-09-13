import os
import logging
import pandas as pd
from datasets import load_from_disk, Dataset, concatenate_datasets
from datetime import datetime
from dataset_loader import load_dual_config_dataset

import argparse
import json

from pipelines import deduplicate_df  

logger = logging.getLogger("dataset_merger")
logging.basicConfig(level=logging.INFO)

DATA_DIR = "preprocess_outputs/en-am/include"


def deduplicate_against_test(
    ds: Dataset,
    test_config: dict,
    src_col: str,
    tgt_col: str,
    logger=None
) -> Dataset:
    """
    Remove rows from ds where (src, tgt) pair exists in the test set.
    """
    # Load test set using config
    test_ds = load_dual_config_dataset(
        name=test_config["name"],
        path=test_config["path"],
        split=test_config["split"],
        src_config=test_config["src_config"],
        tgt_config=test_config["tgt_config"],
        column=test_config["column"],
        format=test_config.get("format", None)
    )
    # Convert both datasets to pandas
    df = ds.to_pandas()
    test_df = test_ds.to_pandas()
    # Build set of (src, tgt) pairs in test set
    test_pairs = set(zip(test_df[src_col], test_df[tgt_col]))
    before = len(df)
    # Remove rows where (src, tgt) in test_pairs
    mask = ~df.apply(lambda row: (row[src_col], row[tgt_col]) in test_pairs, axis=1)
    df = df[mask]
    after = len(df)
    if logger:
        logger.info(f"Removed {before - after} rows found in test set ({after} rows remain)")
    # Convert back to HF dataset
    return Dataset.from_pandas(df, preserve_index=False)

def deduplicate_hf_dataset(
    ds: Dataset,
    src_col: str = "Source",
    tgt_col: str = "Target",
    logger=None
) -> Dataset:
    """
    Deduplicate a Hugging Face Dataset by:
      1. Dropping rows where src == tgt
      2. Dropping duplicate (src, tgt) pairs
      3. Dropping duplicate src values (keep first occurrence)
      4. Dropping duplicate tgt values (keep first occurrence)
    """
    # Convert to pandas
    df = ds.to_pandas()
    before = df.shape[0]

    # (1) Drop rows where src == tgt
    df = df[df[src_col] != df[tgt_col]]
    if logger:
        logger.info(f"Step 1: Drop identical src==tgt rows → {len(df)} rows")

    # (2) Drop duplicate (src, tgt) pairs
    df = df.drop_duplicates(subset=[src_col, tgt_col])
    if logger:
        logger.info(f"Step 2: Drop duplicate (src, tgt) pairs → {len(df)} rows")

    # (3) Drop duplicate sources
    df = df.drop_duplicates(subset=[src_col])
    if logger:
        logger.info(f"Step 3: Drop duplicate sources → {len(df)} rows")

    # (4) Drop duplicate targets
    df = df.drop_duplicates(subset=[tgt_col])
    after = df.shape[0]
    if logger:
        logger.info(f"Step 4: Drop duplicate targets → {after} rows (Removed {before - after})")

    # Convert back to HF dataset
    deduped = Dataset.from_pandas(df, preserve_index=False)
    return deduped


def merge_and_deduplicate_filtered(data_dir, src_col, tgt_col, config, dedup=True, dedup_against_test=True) -> Dataset:
    """
    Merge and deduplicate all datasets in data_dir (no filtering).
    """
    logger.info(f"🔎 Looking for datasets in: {data_dir}")
    datasets = []
    included_datasets = []

    for root, dirs, files in os.walk(data_dir):
        if "metadata.json" in files:
            meta_path = os.path.join(root, "metadata.json")
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                ds = load_from_disk(root)
                logger.info(f"✅ Including {meta.get('dataset_name', root)} → {len(ds)} rows")
                datasets.append(ds)
                included_datasets.append({
                    "dataset_name": meta.get("dataset_name", root),
                    "rows": len(ds),
                    "quality_score": meta.get("quality_score")
                })
            except Exception as e:
                logger.warning(f"⚠ Error reading {meta_path}: {e}")

    if not datasets:
        logger.warning(f"No datasets found for merge in {data_dir}")
        return None

    # Merge all datasets
    
    merged = concatenate_datasets(datasets)
    merged_size = len(merged)
    logger.info(f"📦 Merged dataset size: {merged_size} rows")
    if dedup:
        logger.info("🧹 Starting deduplication process...")
        deduped = deduplicate_hf_dataset(merged, src_col=src_col, tgt_col=tgt_col, logger=logger)
        deduped_size = len(deduped)
        logger.info(f"✨ Deduplicated dataset size: {deduped_size} rows")
        merged = deduped
    else:
        deduped_size = merged_size
        logger.info("⚠ Deduplication skipped as per configuration.")    

    merged_path = os.path.join(
        data_dir, f"merged_{src_col}-{tgt_col}"
    )
    merged.save_to_disk(merged_path)
    logger.info(f"💾 Saved merged dataset → {merged_path}")

    if dedup_against_test:
        logger.info("🧹 Deduplicating against test set...")
        test_config = config.get("test_set")
        if test_config:
            merged = deduplicate_against_test(
                merged,
                test_config=test_config,
                src_col=src_col,
                tgt_col=tgt_col,
                logger=logger
            )
            final_size = len(merged)
            logger.info(f"✨ Final dataset size after test deduplication: {final_size} rows")
        else:
            logger.warning("⚠ No test set configuration found; skipping test deduplication.")

    merged_metadata = {
        "dataset_name": f"merged",
        "lang_pair": f"{src_col}-{tgt_col}",
        "included_datasets": included_datasets,
        "total_rows_before_dedup": merged_size,
        "total_rows_after_dedup": deduped_size,
        "processed_at": datetime.utcnow().isoformat()
    }
    with open(os.path.join(merged_path, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(merged_metadata, f, ensure_ascii=False, indent=2)
    logger.info(f"📝 Metadata written for merged dataset → {os.path.join(merged_path, 'metadata.json')}")

    return merged



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and deduplicate datasets")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="Directory containing datasets to merge")
    parser.add_argument("--src_col", type=str, default="en", help="Source lallnguage column name")
    parser.add_argument("--tgt_col", type=str, required=True, help="Target language column name")
    args = parser.parse_args()
    merged_dataset = merge_and_deduplicate_filtered(
        data_dir=args.data_dir,
        src_col=args.src_col,
        tgt_col=args.tgt_col
    )
    merged_dataset.save_to_disk(f"preprocess_outputs/{args.src_col}-{args.tgt_col}/{args.src_col}-{args.tgt_col}-merged")
    logger.info("Saved merged and deduplicated dataset to preprocess_outputs/en-am-merged")