
# Function to split source lines into chunks to avoid out-of-memory errors
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim
import torch
from tqdm import tqdm
import os
import sys

import pandas as pd
import numpy as np
import re
import csv
import logging

logger = logging.getLogger("my_logger")


def deduplicate_df(df: pd.DataFrame) -> pd.DataFrame:

    df["Source-Copied"] = df['Source'] == df['Target']
    df = df.set_index(['Source-Copied'])

    try:  # Handle case where there are no identical rows
        df = df.drop([True])
    except KeyError:
        pass

    df = df.reset_index()
    df = df.drop(['Source-Copied'], axis=1)

    if logger:
        logger.info(f"Step: Drop identical rows\t--> Rows: {df.shape[0]}")

    return df


def rule_filter(source_texts, target_texts, min_length=3, max_length=200, max_length_ratio=2.0, lower=False):
    logger.debug(f"Source length:{len(source_texts)} Target Legnth:{len(target_texts)}")
    df = pd.DataFrame({"Source": source_texts, "Target": target_texts})
    logger.info(f"Rule filter started: initial rows = {df.shape[0]}")

    # Delete nan
    df = df.dropna()
    logger.info(f"Step: Drop NaN\t\t--> Rows: {df.shape[0]}")

    # Drop duplicates
    df = df.drop_duplicates()
    logger.info(f"Step: Drop duplicates\t--> Rows: {df.shape[0]}")

    # Drop identical rows (moved to helper function)
    df = deduplicate_df(df)

    # Drop too-long rows
    df["Too-Long"] = ((df['Source'].str.count(' ')+1) > (df['Target'].str.count(' ')+1) * max_length_ratio) |  \
                     ((df['Target'].str.count(' ')+1) > (df['Source'].str.count(' ')+1) * max_length_ratio) |  \
                     ((df['Source'].str.count(' ')+1) > max_length) |  \
                     ((df['Target'].str.count(' ')+1) > max_length)

    df = df.set_index(['Too-Long'])
    try:
        df = df.drop([True])
    except KeyError:
        pass
    df = df.reset_index()
    df = df.drop(['Too-Long'], axis=1)
    logger.info(f"Step: Drop too-long\t--> Rows: {df.shape[0]}")

    # Drop too-short rows
    df["Too-Short"] = ((df['Source'].str.len()) <= min_length) |  \
                      ((df['Target'].str.len()) <= min_length)

    df = df.set_index(['Too-Short'])
    try:
        df = df.drop([True])
    except KeyError:
        pass
    df = df.reset_index()
    df = df.drop(['Too-Short'], axis=1)
    logger.info(f"Step: Drop too-short\t--> Rows: {df.shape[0]}")

    # Remove HTML and normalize
    df = df.replace(r'<.*?>|&lt;.*?&gt;|&?(amp|nbsp|quot);|{}', ' ', regex=True)
    df = df.replace(r'  ', ' ', regex=True)
    logger.info(f"Step: Clean HTML\t--> Rows: {df.shape[0]}")

    # Lower-case if requested
    if lower:
        df['Source'] = df['Source'].str.lower()
        df['Target'] = df['Target'].str.lower()
        logger.info("Step: Lowercased rows")
    else:
        logger.info("Step: Truecased rows retained")

    # Replace empty cells with NaN, then drop them
    df = df.replace(r'^\s*$', np.nan, regex=True)
    logger.info(f"Step: Drop new NaNs\t--> Rows: {df.shape[0]}")
    df = df.dropna()

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    logger.info(f"Step: Shuffled rows\t--> Rows: {df.shape[0]}")

    return df["Source"].tolist(), df["Target"].tolist()


def semantic_filter(
    source_list,
    target_list,
    srclang,
    tgtlang,
    threshold=0.7,
    chunk_size=1000,
    batch_size=2048,
    model=None,
    pool=None
    ):
    assert len(source_list) == len(target_list), "Source and target lists must be of the same length."
    
    logger.info("Semantic filter started")
    logger.info(f"Total sentence pairs: {len(source_list)} | Threshold: {threshold} | Chunk size: {chunk_size}")

    # model = load_model(srclang, tgtlang)
    # pool = model.start_multi_process_pool()

    filtered_source = []
    filtered_target = []


    for i in range(0, len(source_list), chunk_size):
        end_idx = min(i + chunk_size, len(source_list))
        logger.info(f"Processing chunk: lines {i}–{end_idx}")

        chunk_src = source_list[i:i + chunk_size]
        chunk_tgt = target_list[i:i + chunk_size]

        # Encode using Sentence Transformer
        source_embeddings = model.encode(chunk_src, pool=pool, batch_size=batch_size)
        target_embeddings = model.encode(chunk_tgt, pool=pool, batch_size=batch_size)

        for src_text, tgt_text, src_vec, tgt_vec in zip(chunk_src, chunk_tgt, source_embeddings, target_embeddings):
            similarity = pytorch_cos_sim(src_vec, tgt_vec).item()
            if similarity > threshold:
                filtered_source.append(src_text)
                filtered_target.append(tgt_text)


    logger.info(f"Semantic filtering complete → Remaining: {len(filtered_source)} pairs")
    return filtered_source, filtered_target

    
def lang_detect_filter(
    source_list,
    target_list,
    srclang,
    tgtlang,
    model,
    batch_size=1024,
    min_score=0.9
):
    
    assert len(source_list) == len(target_list), "Source and target lists must be of the same length."
    logger.info("Language detection filter started")
    logger.info(f"Total sentence pairs: {len(source_list)} | Batch size: {batch_size} | Min score: {min_score}")

    def detect(lines):
        results, scores = [], []
        for i in range(0, len(lines), batch_size):
            batch = lines[i:i+batch_size]
            predictions = model.predict(batch, k=1)
            codes = [pred[0].replace("__label__", "") for pred in predictions[0]]
            scs   = [pred[0] for pred in predictions[1]]
            results.extend(codes)
            scores.extend(scs)
        return results, scores

    source_list = [s.replace("\n", " ") for s in source_list]
    target_list = [t.replace("\n", " ") for t in target_list]
    # Detect languages
    src_codes, src_scores = detect(source_list)
    tgt_codes, tgt_scores = detect(target_list)

    filtered_source = []
    filtered_target = []

    for s, t, sl, tl, ss, ts in zip(source_list, target_list, src_codes, tgt_codes, src_scores, tgt_scores):
        if sl == srclang and tl == tgtlang and ss >= min_score and ts >= min_score:
            filtered_source.append(s)
            filtered_target.append(t)

    logger.info(f"Language detection complete → Remaining: {len(filtered_source)} pairs")
    return filtered_source, filtered_target


if __name__=="__main__":
    from datasets import load_dataset
    ds = load_dataset("google/smol", "smolsent__en_am")
    source_txts = ds["train"]['src']
    target_txts = ds["train"]['trg']
    print(f"Length of source files:{len(source_txts)} \n Length target files:{len(target_txts)}")
    semantic_filter(source_txts, target_txts, "en", "fa")

