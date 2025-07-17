
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

# display(df) works only if you are in IPython/Jupyter Notebooks or enable:
#from IPython.display import display


def rule_filter(source_texts, target_texts, min_length=3, max_length=200, max_length_ratio=2.0, lower=False):
    
    df = pd.DataFrame({"Source": source_texts, "Target": target_texts})
    logger.info(f"Rule filter started: initial rows = {df.shape[0]}")
 
    # Delete nan
    df = df.dropna()
    logger.info(f"Step: Drop NaN\t\t--> Rows: {df.shape[0]}")


    # Drop duplicates
    df = df.drop_duplicates()
    logger.info(f"Step: Drop duplicates\t--> Rows: {df.shape[0]}")

    # Drop copy-source rows
    df["Source-Copied"] = df['Source'] == df['Target']
    #display(df.loc[df['Source-Copied'] == True]) # display only copy-sourced rows
    df = df.set_index(['Source-Copied'])

    try: # To avoid (KeyError: '[True] not found in axis') if there are no source-copied cells
        df = df.drop([True]) # Boolean, not string, do not add quotes
    except:
        pass
    
    df = df.reset_index()
    df = df.drop(['Source-Copied'], axis = 1)
    logger.info(f"Step: Drop identical rows\t--> Rows: {df.shape[0]}")


    # Drop too-long rows (source or target)
    # Based on your language, change the values "2" and "200"
    df["Too-Long"] = ((df['Source'].str.count(' ')+1) > (df['Target'].str.count(' ')+1) * max_length_ratio) |  \
                     ((df['Target'].str.count(' ')+1) > (df['Source'].str.count(' ')+1) * max_length_ratio) |  \
                     ((df['Source'].str.count(' ')+1) > max_length) |  \
                     ((df['Target'].str.count(' ')+1) > max_length)
                
    #display(df.loc[df['Too-Long'] == True]) # display only too long rows
    df = df.set_index(['Too-Long'])

    try: # To avoid (KeyError: '[True] not found in axis') if there are no too-long cells
        df = df.drop([True]) # Boolean, not string, do not add quotes
    except:
        pass

    df = df.reset_index()
    df = df.drop(['Too-Long'], axis = 1)
    logger.info(f"Step: Drop too-long\t--> Rows: {df.shape[0]}")


    # Drop too-short rows (source or target)
    # Based on your language, change the values "5"
    df["Too-Short"] = ((df['Source'].str.len()) <= min_length) |  \
                      ((df['Target'].str.len()) <= min_length)
                
    df = df.set_index(['Too-Short'])

    try: # To avoid (KeyError: '[True] not found in axis') if there are no too-long cells
        df = df.drop([True]) # Boolean, not string, do not add quotes
    except:
        pass

    df = df.reset_index()
    df = df.drop(['Too-Short'], axis = 1)
    logger.info(f"Step: Drop too-short\t--> Rows: {df.shape[0]}")


    # Remove HTML and normalize
    # Use str() to avoid (TypeError: expected string or bytes-like object)
    # Note: removing tags should be before removing empty cells because some cells might have only tags and become empty.

    df = df.replace(r'<.*?>|&lt;.*?&gt;|&?(amp|nbsp|quot);|{}', ' ', regex=True)
    df = df.replace(r'  ', ' ', regex=True)  # replace double-spaces with one space
    logger.info(f"Step: Clean HTML\t--> Rows: {df.shape[0]}")


    # Lower-case the data
    if lower == True:
        df['Source'] = df['Source'].str.lower()
        df['Target'] = df['Target'].str.lower()
        logger.info("Step: Lowercased rows")
    else:
        logger.info("Step: Truecased rows retained")


    # Replace empty cells with NaN
    df = df.replace(r'^\s*$', np.nan, regex=True)
    logger.info(f"Step: Drop new NaNs\t--> Rows: {df.shape[0]}")

    # Delete nan (already there, or generated from the previous steps)
    df = df.dropna()
    logger.info(f"Step: Shuffled rows\t--> Rows: {df.shape[0]}")


    # Shuffle the data
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
):
    assert len(source_list) == len(target_list), "Source and target lists must be of the same length."
    
    logger.info("Semantic filter started")
    logger.info(f"Total sentence pairs: {len(source_list)} | Threshold: {threshold} | Chunk size: {chunk_size}")

    model = load_model(srclang, tgtlang)
    pool = model.start_multi_process_pool()

    filtered_source = []
    filtered_target = []

    for i in range(0, len(source_list), chunk_size):
        end_idx = min(i + chunk_size, len(source_list))
        logger.info(f"Processing chunk: lines {i}–{end_idx}")

        chunk_src = source_list[i:i + chunk_size]
        chunk_tgt = target_list[i:i + chunk_size]

        # Encode using Sentence Transformer
        source_embeddings = model.encode_multi_process(chunk_src, pool=pool, batch_size=batch_size)
        target_embeddings = model.encode_multi_process(chunk_tgt, pool=pool, batch_size=batch_size)

        for src_text, tgt_text, src_vec, tgt_vec in zip(chunk_src, chunk_tgt, source_embeddings, target_embeddings):
            similarity = pytorch_cos_sim(src_vec, tgt_vec).item()
            if similarity > threshold:
                filtered_source.append(src_text)
                filtered_target.append(tgt_text)

    model.stop_multi_process_pool(pool)
    logger.info(f"Semantic filtering complete → Remaining: {len(filtered_source)} pairs")
    return filtered_source, filtered_target

    


# def line_count(filename):
#     f = open(filename, 'rb')
#     lines = 0
#     buf_size = 1024 * 1024
#     read_f = f.raw.read

#     buf = read_f(buf_size)
#     while buf:
#         lines += buf.count(b'\n')
#         buf = read_f(buf_size)

#     return lines

def load_model(srclang, tgtlang):
    # Download and load the model
    model_cache = "model_cache"
    
    muse_langs = ['ar', 'de', 'en', 'es', 'fr', 'it', 'ko', 'nl', 'pt', 'pt', 'ru', 'tr', 'zh']
    para_langs = ["ar", "bg", "ca", "cs", "da", "de", "en", "el", "es", "et", "fa", "fi", "fr", "gl", "gu", "he", "hi", "hr", "hu", "hy", "id", "it", "ja", "ka", "ko", "ku", "lt", "lv", "mk", "mn", "mr", "ms", "my", "nb", "nl", "pl", "pt", "pt", "ro", "ru", "sk", "sl", "sq", "sr", "sv", "th", "tr", "uk", "ur", "vi", "zh"]
    microsoft_langs = ["en", "ar", "bg", "de", "el", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]
    labse_codes = [
    "af", "sq", "am", "ar", "hy", "as", "az", "eu", "be", "bn", "bs", "bg", "my", "ca", "ceb",
    "zh", "co", "hr", "cs", "da", "nl", "en", "eo", "et", "fi", "fr", "fy", "gl", "ka", "de",
    "el", "gu", "ht", "ha", "haw", "he", "hi", "hmn", "hu", "is", "ig", "id", "ga", "it", "ja",
    "jv", "kn", "kk", "km", "rw", "ko", "ku", "ky", "lo", "la", "lv", "lt", "lb", "mk", "mg",
    "ms", "ml", "mt", "mi", "mr", "mn", "ne", "no", "ny", "or", "fa", "pl", "pt", "pa", "ro",
    "ru", "sm", "gd", "sr", "st", "sn", "si", "sk", "sl", "so", "es", "su", "sw", "sv", "tl",
    "tg", "ta", "tt", "te", "th", "bo", "tr", "tk", "ug", "uk", "ur", "uz", "vi", "cy", "wo",
    "xh", "yi", "yo", "zu"
]

    
    if len(srclang) > 2 or len(tgtlang) > 2:
        raise SystemExit("Please use an ISO 639‑1 language code, e.g. 'en'!")
    elif srclang in muse_langs and tgtlang in muse_langs:
        model_name = "distiluse-base-multilingual-cased-v1"  # 15 languages
    elif srclang in para_langs and tgtlang in para_langs:
        model_name = "paraphrase-multilingual-MiniLM-L12-v2"  # 50 languages
    elif srclang in microsoft_langs and tgtlang in microsoft_langs:
        model_name = "microsoft/Multilingual-MiniLM-L12-H384"  # 16 language
    elif srclang in labse_codes and tgtlang in labse_codes:
        model_name = "sentence-transformers/LaBSE"
    else:
        raise SystemExit("Language pair is not supported!")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name, device=device, cache_folder=model_cache)
    logger.info(f"Loaded SentenceTransformer model: {model_name} on {device}")
    pool = model.start_multi_process_pool()

    return model


if __name__=="__main__":
    from datasets import load_dataset
    ds = load_dataset("google/smol", "smolsent__en_am")
    source_txts = ds["train"]['src']
    target_txts = ds["train"]['trg']
    print(f"Length of source files:{len(source_txts)} \n Length target files:{len(target_txts)}")
    semantic_filter(source_txts, target_txts, "en", "fa")

