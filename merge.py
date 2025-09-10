import os
import logging
import pandas as pd
from datasets import load_from_disk, Dataset, concatenate_datasets
from datetime import datetime

import argparse
import json

from pipelines import deduplicate_df  

logger = logging.getLogger("dataset_merger")
logging.basicConfig(level=logging.INFO)

DATA_DIR = "preprocess_outputs/en-am/include"


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


def merge_and_deduplicate_filtered(data_dir, qe_min_score, config, logger, src_col, tgt_col) -> Dataset:
    """
    Merge and deduplicate datasets in data_dir,
    but only include datasets with quality_score >= qe_min_score.
    """
    logger.info(f"🔎 Looking for datasets in: {data_dir}")
    datasets = []
    included_datasets = []
    excluded_datasets = []

    for root, dirs, files in os.walk(data_dir):
        if "metadata.json" in files:
            meta_path = os.path.join(root, "metadata.json")
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                quality = meta.get("quality_score")
                if quality is not None and quality >= qe_min_score:
                    ds = load_from_disk(root)
                    logger.info(f"✅ Including {meta['dataset_name']} → {len(ds)} rows (QE={quality:.2f})")
                    datasets.append(ds)
                    included_datasets.append({
                        "dataset_name": meta["dataset_name"],
                        "rows": len(ds),
                        "quality_score": quality
                    })
                else:
                    logger.info(f"⏭ Skipping {meta['dataset_name']} (QE={quality})")
                    excluded_datasets.append({
                        "dataset_name": meta.get("dataset_name"),
                        "quality_score": quality
                    })
            except Exception as e:
                logger.warning(f"⚠ Error reading {meta_path}: {e}")

    if not datasets:
        logger.warning(f"No datasets qualified for merge in {data_dir}")
        return None

    # Merge all datasets
    merged = concatenate_datasets(datasets)
    merged_size = len(merged)
    logger.info(f"📦 Merged dataset size: {len(merged)} rows")

    # # Convert to pandas for deduplication
    # df = merged.to_pandas()[[src_col, tgt_col]]
    # df = df.rename(columns={src_col: "Source", tgt_col: "Target"})
    # df = deduplicate_df(df)
    # df = df.rename(columns={"Source": src_col, "Target": tgt_col})

    # # Convert back to HF dataset
    # deduped = Dataset.from_pandas(df, preserve_index=False)
    # deduped_size = len(deduped)
    # logger.info(f"✨ Deduplicated dataset size: {len(deduped)} rows")

    deduped = deduplicate_hf_dataset(merged, src_col=src_col, tgt_col=tgt_col, logger=logger)
    deduped_size = len(deduped)
    logger.info(f"✨ Deduplicated dataset size: {len(deduped)} rows")

    merged_path = os.path.join(
        data_dir, f"merged_{src_col}-{tgt_col}_qe{qe_min_score}"
    )
    deduped.save_to_disk(merged_path)
    logger.info(f"💾 Saved merged dataset → {merged_path}")

    
    merged_metadata = {
        "dataset_name": f"merged_qe{qe_min_score}",
        "lang_pair": f"{src_col}-{tgt_col}",
        "included_datasets": included_datasets,
        "excluded_datasets": excluded_datasets,
        "total_rows_before_dedup": merged_size,
        "total_rows_after_dedup": deduped_size,
        "qe_min_score": qe_min_score,
        "processed_at": datetime.utcnow().isoformat()
    }
    with open(os.path.join(merged_path, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(merged_metadata, f, ensure_ascii=False, indent=2)
    logger.info(f"📝 Metadata written for merged dataset → {os.path.join(merged_path, 'metadata.json')}")

    return deduped



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and deduplicate datasets")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="Directory containing datasets to merge")
    parser.add_argument("--src_col", type=str, default="en", help="Source lallnguage column name")
    parser.add_argument("--tgt_col", type=str, required=True, help="Target language column name")
    args = parser.parse_args()
    merged_dataset = merge_and_deduplicate(
        data_dir=args.data_dir,
        src_col=args.src_col,
        tgt_col=args.tgt_col
    )
    merged_dataset.save_to_disk(f"preprocess_outputs/{args.src_col}-{args.tgt_col}/{args.src_col}-{args.tgt_col}-merged")
    logger.info("Saved merged and deduplicated dataset to preprocess_outputs/en-am-merged")