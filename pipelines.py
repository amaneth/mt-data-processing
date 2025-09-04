
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

    
from langdetect import detect as detect_lang
from langdetect import DetectorFactory

DetectorFactory.seed = 0

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

    # AfroLID model and label mapping for supported African languages
    afrolid_model = None
    afrolid_label_map = {
        'am': 'amh',  # Amharic
        'so': 'som',  # Somali
        'sw': 'swh',  # Swahili
        'af': 'afr',  # Afrikaans
        'ha': 'hau',  # Hausa
        'zu': 'zul'   # Zulu
    }

    african_langs = list(afrolid_label_map.keys())

    # Check if we should use AfroLID for better African language detection
    use_afrolid_src = srclang in african_langs
    use_afrolid_tgt = tgtlang in african_langs

    if use_afrolid_src or use_afrolid_tgt:
        logger.info("Using AfroLID for African language detection.")
        try:
            from transformers import pipeline
            afrolid_model = pipeline("text-classification", model='UBC-NLP/afrolid_1.5')
            logger.info("✅ AfroLID model loaded successfully")
        except Exception as e:
            logger.warning(f"❌ Failed to load AfroLID model: {e}, falling back to default")
            afrolid_model = None

    if srclang == 'so' or tgtlang == 'so':
        if afrolid_model and (use_afrolid_src or use_afrolid_tgt):
            logger.info("Using AfroLID for Somali language detection.")
            filtered_source = []
            filtered_target = []

            # Map language codes to AfroLID labels
            target_src_lang = afrolid_label_map.get(srclang, srclang.lower())
            target_tgt_lang = afrolid_label_map.get(tgtlang, tgtlang.lower())

            for s, t in tqdm(zip(source_list, target_list), total=len(source_list)):
                try:
                    s_result = afrolid_model(s) if s.strip() else None
                    t_result = afrolid_model(t) if t.strip() else None

                    if s_result:
                        s_lang = s_result[0]['label'].lower()
                        s_score = s_result[0]['score']
                    else:
                        s_lang, s_score = "", 0

                    if t_result:
                        t_lang = t_result[0]['label'].lower()
                        t_score = t_result[0]['score']
                    else:
                        t_lang, t_score = "", 0

                    # Check if detected languages match and scores are above threshold
                    if (s_lang == target_src_lang and t_lang == target_tgt_lang and
                        s_score >= min_score and t_score >= min_score):
                        filtered_source.append(s)
                        filtered_target.append(t)
                except Exception as e:
                    logger.debug(f"AfroLID detection error: {e}")
                    continue

            logger.info(f"Language detection with AfroLID complete → Remaining: {len(filtered_source)} pairs")
            return filtered_source, filtered_target

        logger.info("Using langdetect for Somali language detection.")
        filtered_source = []
        filtered_target = []
        for s, t in tqdm(zip(source_list, target_list), total=len(source_list)):
            try:
                s_lang = detect_lang(s)
                t_lang = detect_lang(t)
                if s_lang == srclang and t_lang == tgtlang:
                    filtered_source.append(s)
                    filtered_target.append(t)
            except Exception:
                continue
        logger.info(f"Language detection complete → Remaining: {len(filtered_source)} pairs")
        return filtered_source, filtered_target

    def detect(lines, lang_code=None):
        results, scores = [], []

        # Use AfroLID if available and language is African
        if afrolid_model and lang_code in african_langs:
            logger.info(f"Using AfroLID for detecting {lang_code}")
            target_lang = afrolid_label_map.get(lang_code, lang_code.lower())

            for text in tqdm(lines, total=len(lines)):
                try:
                    if text.strip():
                        result = afrolid_model(text)
                        det_lang = result[0]['label'].lower()
                        det_score = result[0]['score']

                        # Map AfroLID labels back to expected format
                        reverse_map = {v: k for k, v in afrolid_label_map.items()}
                        mapped_lang = reverse_map.get(det_lang, det_lang)
                        results.append(mapped_lang)
                        scores.append(det_score)
                    else:
                        results.append('')
                        scores.append(0.0)
                except Exception as e:
                    logger.debug(f"AfroLID detection error: {e}")
                    results.append('')
                    scores.append(0.0)
        else:
            # Fall back to FastText
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
    src_codes, src_scores = detect(source_list, srclang)
    tgt_codes, tgt_scores = detect(target_list, tgtlang)

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

