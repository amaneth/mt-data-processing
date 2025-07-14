
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


# display(df) works only if you are in IPython/Jupyter Notebooks or enable:
#from IPython.display import display


def rule_filter(source_texts, target_texts, source_lang, target_lang, lower=False):
    
    # df_source = pd.read_csv(source_file, names=['Source'], sep="\0", quoting=csv.QUOTE_NONE, skip_blank_lines=False, on_bad_lines="skip")
    # df_target = pd.read_csv(target_file, names=['Target'], sep="\0", quoting=csv.QUOTE_NONE, skip_blank_lines=False, on_bad_lines="skip")
    # df = pd.concat([df_source, df_target], axis=1)  # Join the two dataframes along columns
    # print("Dataframe shape (rows, columns):", df.shape)

    df = pd.DataFrame({"Source": source_texts, "Target": target_texts})



    
    # Delete nan
    df = df.dropna()

    print("--- Rows with Empty Cells Deleted\t--> Rows:", df.shape[0])


    # Drop duplicates
    df = df.drop_duplicates()
    #df = df.drop_duplicates(subset=['Target'])

    print("--- Duplicates Deleted\t\t\t--> Rows:", df.shape[0])


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
    
    print("--- Source-Copied Rows Deleted\t\t--> Rows:", df.shape[0])


    # Drop too-long rows (source or target)
    # Based on your language, change the values "2" and "200"
    df["Too-Long"] = ((df['Source'].str.count(' ')+1) > (df['Target'].str.count(' ')+1) * 2) |  \
                     ((df['Target'].str.count(' ')+1) > (df['Source'].str.count(' ')+1) * 2) |  \
                     ((df['Source'].str.count(' ')+1) > 200) |  \
                     ((df['Target'].str.count(' ')+1) > 200)
                
    #display(df.loc[df['Too-Long'] == True]) # display only too long rows
    df = df.set_index(['Too-Long'])

    try: # To avoid (KeyError: '[True] not found in axis') if there are no too-long cells
        df = df.drop([True]) # Boolean, not string, do not add quotes
    except:
        pass

    df = df.reset_index()
    df = df.drop(['Too-Long'], axis = 1)

    print("--- Too Long Source/Target Deleted\t--> Rows:", df.shape[0])


    # Drop too-short rows (source or target)
    # Based on your language, change the values "5"
    df["Too-Short"] = ((df['Source'].str.len()) <= 3) |  \
                      ((df['Target'].str.len()) <= 3)
                
    df = df.set_index(['Too-Short'])

    try: # To avoid (KeyError: '[True] not found in axis') if there are no too-long cells
        df = df.drop([True]) # Boolean, not string, do not add quotes
    except:
        pass

    df = df.reset_index()
    df = df.drop(['Too-Short'], axis = 1)

    print("--- Too Short Source/Target Deleted\t--> Rows:", df.shape[0])


    # Remove HTML and normalize
    # Use str() to avoid (TypeError: expected string or bytes-like object)
    # Note: removing tags should be before removing empty cells because some cells might have only tags and become empty.

    df = df.replace(r'<.*?>|&lt;.*?&gt;|&?(amp|nbsp|quot);|{}', ' ', regex=True)
    df = df.replace(r'  ', ' ', regex=True)  # replace double-spaces with one space

    print("--- HTML Removed\t\t\t--> Rows:", df.shape[0])


    # Lower-case the data
    if lower == True:
        df['Source'] = df['Source'].str.lower()
        df['Target'] = df['Target'].str.lower()

        print("--- Rows are now lower-cased\t\t--> Rows:", df.shape[0])
    else:
        print("--- Rows will remain true-cased\t\t--> Rows:", df.shape[0])


    # Replace empty cells with NaN
    df = df.replace(r'^\s*$', np.nan, regex=True)

    # Delete nan (already there, or generated from the previous steps)
    df = df.dropna()

    print("--- Rows with Empty Cells Deleted\t--> Rows:", df.shape[0])


    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    print("--- Rows Shuffled\t\t\t--> Rows:", df.shape[0])


    # Write the dataframe to two Source and Target files
    # source_file = source_file+'-filtered.'+source_lang
    # target_file = target_file+'-filtered.'+target_lang


    # Save source and target to two text files
    # df_source = df["Source"]
    # df_target = df["Target"]

    # df_source.to_csv(source_file, header=False, index=False, quoting=csv.QUOTE_NONE, sep="\n")
    # print("--- Source Saved:", source_file)
    # df_target.to_csv(target_file, header=False, index=False, quoting=csv.QUOTE_NONE, sep="\n")
    # print("--- Target Saved:", target_file)

    return df["Source"].tolist(), df["Target"].tolist()


def semantic_filter(
    source_list,
    target_list,
    srclang,
    tgtlang,
    threshold=0.7,
    chunk_size=1000,
):
    assert len(source_list) == len(target_list), "Source and target lists must be of the same length."

    model = load_model(srclang, tgtlang)
    pool = model.start_multi_process_pool()

    filtered_source = []
    filtered_target = []

    for i in range(0, len(source_list), chunk_size):
        print(f"Processing lines {i}–{min(i + chunk_size, len(source_list))} ...", flush=True)

        chunk_src = source_list[i:i + chunk_size]
        chunk_tgt = target_list[i:i + chunk_size]

        # Encode using Sentence Transformer
        source_embeddings = model.encode_multi_process(chunk_src, pool=pool, batch_size=2048)
        target_embeddings = model.encode_multi_process(chunk_tgt, pool=pool, batch_size=2048)

        for src_text, tgt_text, src_vec, tgt_vec in zip(chunk_src, chunk_tgt, source_embeddings, target_embeddings):
            similarity = pytorch_cos_sim(src_vec, tgt_vec).item()
            if similarity > threshold:
                filtered_source.append(src_text)
                filtered_target.append(tgt_text)

    model.stop_multi_process_pool(pool)
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
    
    if len(srclang) > 2 or len(tgtlang) > 2:
        raise SystemExit("Please use an ISO 639‑1 language code, e.g. 'en'!")
    elif srclang in muse_langs and tgtlang in muse_langs:
        model_name = "distiluse-base-multilingual-cased-v1"  # 15 languages
    elif srclang in para_langs and tgtlang in para_langs:
        model_name = "paraphrase-multilingual-MiniLM-L12-v2"  # 50 languages
    else:
        raise SystemExit("Language pair is not supported!")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name, device=device, cache_folder=model_cache)
    print("Model loaded:", model_name) 
    pool = model.start_multi_process_pool()

    return model


if __name__=="__main__":
    from datasets import load_dataset
    ds = load_dataset("google/smol", "smolsent__en_am")
    source_txts = ds["train"]['src']
    target_txts = ds["train"]['trg']
    print(f"Length of source files:{len(source_txts)} \n Length target files:{len(target_txts)}")
    semantic_filter(source_txts, target_txts, "en", "fa")

