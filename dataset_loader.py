import os
from datasets import load_dataset
from fetch import download_url, download_opus, download_table

def load_flat_column_dataset(cfg, dataset_cache=None):

    trust_remote_code = cfg.get("trust_remote_code", False)
    auth_token = cfg.get("auth_token")
    kwargs = {} if auth_token is None else {"token": auth_token}
    kwargs["trust_remote_code"] = trust_remote_code

    if "config_name" in cfg:
        dataset = load_dataset(cfg["path"], cfg["config_name"], split=cfg["split"], cache_dir=dataset_cache, **kwargs)
    else:
        dataset = load_dataset(cfg["path"], split=cfg["split"], cache_dir=dataset_cache, **kwargs)
    if cfg["path"] == "refine-ai/subscene":
        # Special handling for subscene dataset
        source_list = []
        target_list = []
        for item in dataset:
            if 'transcript' in item:
                text_list = [t['text'] for t in item['transcript'] if 'text' in t]
                full_text = ' '.join(text_list)
                source_list.append(full_text)
                target_list.append(full_text)  # For now, duplicate since subscene may not be paired
        return source_list, target_list

    # Normal flat column processing
    source_list = [item[cfg["src_col"]] for item in dataset]
    target_list = [item[cfg["tgt_col"]] for item in dataset]


    return source_list, target_list
def load_dual_config_dataset(cfg, dataset_cache=None):

    trust_remote_code = cfg.get("trust_remote_code", False)
    auth_token = cfg.get("auth_token")
    kwargs = {} if auth_token is None else {"token": auth_token}
    kwargs["trust_remote_code"] = trust_remote_code

    source_list = load_dataset(cfg["path"], cfg["src_config"], split=cfg["split"], cache_dir=dataset_cache, **kwargs)[cfg["column"]]
    target_list = load_dataset(cfg["path"], cfg["tgt_config"], split=cfg["split"], cache_dir=dataset_cache, **kwargs)[cfg["column"]]

    return source_list, target_list

def load_nested_translation_dataset(cfg, dataset_cache=None):
    if cfg.get("path") == "IWSLT/ted_talks_iwslt":
        # Special handling for ted_talks_iwslt dataset which requires language_pair and year parameters
        total_source = []
        total_target = []
        for year in ['2014', '2015', '2016']:
            try:
                dataset = load_dataset(cfg["path"], language_pair=("en", "sw"), year=year, split="train", cache_dir=dataset_cache)
                column = cfg.get("column", "translation")
                for trans in dataset[column]:
                    if "en" in trans and "sw" in trans:
                        total_source.append(trans["en"])
                        total_target.append(trans["sw"])
            except Exception as e:
                print(f"Year {year} for IWSLT/ted_talks_iwslt failed: {e}")
                continue
        return total_source, total_target
    else:
        trust_remote_code = cfg.get("trust_remote_code", False)
        auth_token = cfg.get("auth_token")
        kwargs = {} if auth_token is None else {"token": auth_token}
        kwargs["trust_remote_code"] = trust_remote_code

        if "config_name" in cfg:
            dataset = load_dataset(cfg["path"], cfg["config_name"], split=cfg["split"], cache_dir=dataset_cache, **kwargs)
        else:
            dataset = load_dataset(cfg["path"], split=cfg["split"], cache_dir=dataset_cache, **kwargs)

        column = cfg.get("column", "translation")
        source_list = [trans[cfg["src_key"]] for trans in dataset[column]]
        target_list = [trans[cfg["tgt_key"]] for trans in dataset[column]]

    return source_list, target_list



def load_hf_opus_dataset(cfg, dataset_cache=None):
    """Load OPUS dataset from Hugging Face using lang1 and lang2 parameters"""
    trust_remote_code = cfg.get("trust_remote_code", False)
    auth_token = cfg.get("auth_token")
    kwargs = {} if auth_token is None else {"token": auth_token}
    kwargs["trust_remote_code"] = trust_remote_code

    dataset = load_dataset(cfg["path"], lang1=cfg["lang1"], lang2=cfg["lang2"], split="train", cache_dir=dataset_cache, **kwargs)

    # OPUS datasets typically have a 'translation' column with nested structure
    source_list = [item["translation"][cfg["lang1"]] for item in dataset]
    target_list = [item["translation"][cfg["lang2"]] for item in dataset]

    return source_list, target_list

def load_hf_dataset(cfg, dataset_cache=None):
    format = cfg.get("format", "flat_column")
    if format == "flat_column":
        return load_flat_column_dataset(cfg, dataset_cache)
    elif format == "nested_translation":
        return load_nested_translation_dataset(cfg, dataset_cache)
    elif format == "dual_config":
        return load_dual_config_dataset(cfg, dataset_cache)
    elif format == "hf_opus":
        return load_hf_opus_dataset(cfg, dataset_cache)
    else:
        raise ValueError(f"Unsupported dataset format: {format}")
def load_url_dataset(cfg, srclang, tgtlang, raw_dir=None):
    name = cfg["name"]
    src_path, tgt_path = download_url(
        cfg["src_url"],
        cfg["tgt_url"],
        src_name=f"{name}.{srclang}",
        tgt_name=f"{name}.{tgtlang}",
        base_dir=raw_dir,
    )
    with open(src_path, "r") as f:
        source_list = f.read().splitlines()
    with open(tgt_path, "r") as f:
        target_list = f.read().splitlines()
    
    return source_list, target_list
def load_opus_dataset(cfg, srclang, tgtlang, raw_dir=None, dataset_cache=None):
    url = cfg["url"]
    name = cfg["name"]

    print(f"Debug: Loading OPUS dataset '{name}' for {srclang}-{tgtlang}")
    source_file, target_file = download_opus(srclang, tgtlang, name, url, base_dir=raw_dir)

    source_path = os.path.join(raw_dir, srclang+"-"+tgtlang, source_file)
    target_path = os.path.join(raw_dir, srclang+"-"+tgtlang, target_file)

    with open(source_path, "r", encoding="utf-8", errors="replace") as f:
        source_list = f.read().splitlines()
    with open(target_path, "r", encoding="utf-8", errors="replace") as f:
        target_list = f.read().splitlines()
    
        # Handle length mismatch by keeping only paired sentences
        min_lines = min(len(source_list), len(target_list))
        if len(source_list) != len(target_list):
            discarded_count = abs(len(source_list) - len(target_list))
    
        source_list = source_list[:min_lines]
    target_list = target_list[:min_lines]

    return source_list, target_list

def load_table_dataset(cfg, raw_dir=None):
    import pandas as pd

    # Download the file
    file_path = download_table(
        url=cfg["url"],
        save_name=cfg["name"],
        file_type=cfg.get("file_type", "tsv"),
        base_dir=raw_dir,
    )

    # Load with pandas
    if cfg.get("file_type", "tsv") == "tsv":
        df = pd.read_csv(
                    file_path,
                    sep="\t",
                    quoting=3,  # csv.QUOTE_NONE
                    on_bad_lines='skip',
                    engine="python"
                )
    else:
        df = pd.read_csv(file_path)

    # Extract source + target as lists
    source_list = df[cfg["src_col"]].astype(str).tolist()
    target_list = df[cfg["tgt_col"]].astype(str).tolist()

    return source_list, target_list

def load_dataset_standard(cfg, srclang, tgtlang, raw_dir=None, dataset_cache=None):
    source = cfg.get("source", "hf")
    if source == "hf":
        return load_hf_dataset(cfg, dataset_cache)
    elif source =="opus":
        return load_opus_dataset(cfg, srclang, tgtlang, raw_dir)
    elif source == "url":
        return load_url_dataset(cfg, srclang, tgtlang, raw_dir)
    elif source == "table":
        return load_table_dataset(cfg, raw_dir)
    else:
        raise NotImplementedError(f"Unknown source: {source}")
