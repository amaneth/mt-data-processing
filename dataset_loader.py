import os
from datasets import load_dataset
from fetch import download_url, download_opus, download_table

def load_flat_column_dataset(cfg, dataset_cache=None, filter_param=None):
    extra_args = cfg.get("extra_args", {})
    if dataset_cache:
        extra_args["cache_dir"] = dataset_cache

    # Load dataset (with or without config)
    if "config_name" in cfg:
        dataset = load_dataset(cfg["path"], cfg["config_name"], split=cfg["split"], **extra_args)
    else:
        dataset = load_dataset(cfg["path"], split=cfg["split"], **extra_args)

    # Optional filtering
    if filter_param:
        key, value = next(iter(filter_param.items()))
        dataset = dataset.filter(lambda example: example.get(key) == value)

    # Extract columns
    source_list = [item[cfg["src_col"]] for item in dataset]
    target_list = [item[cfg["tgt_col"]] for item in dataset]
    
    return source_list, target_list

    
def load_dual_config_dataset(cfg, dataset_cache=None):
    
    source_list = load_dataset(cfg["path"], cfg["src_config"], split=cfg["split"], cache_dir=dataset_cache)[cfg["column"]]
    target_list = load_dataset(cfg["path"], cfg["tgt_config"], split=cfg["split"], cache_dir=dataset_cache)[cfg["column"]]

    return source_list, target_list

def load_nested_translation_dataset(cfg, dataset_cache=None):
    extra_args = cfg.get("extra_args", {})
    if dataset_cache:
        extra_args["cache_dir"] = dataset_cache

    dataset = load_dataset(cfg["path"], cfg.get("config_name"), split=cfg["split"], **extra_args)
    source_list = [trans[cfg["src_key"]] for trans in dataset[cfg["column"]]]
    target_list = [trans[cfg["tgt_key"]] for trans in dataset[cfg["column"]]]
    return source_list, target_list



def load_hf_dataset(cfg, dataset_cache=None):
    format = cfg.get("format", "flat_column")
    if format == "flat_column":
        return load_flat_column_dataset(cfg, dataset_cache)
    elif format == "nested_translation":
        return load_nested_translation_dataset(cfg, dataset_cache)
    elif format == "dual_config":
        return load_dual_config_dataset(cfg, dataset_cache)
    else:
        raise ValueError(f"Unsupported dataset format: {format}")
def load_url_dataset(cfg, srclang, tgtlang, raw_dir=None):
    name = cfg["name"]
    print(raw_dir)
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
    source_file, target_file = download_opus(srclang, tgtlang, name, url, base_dir=raw_dir)
    with open(os.path.join(raw_dir, srclang+"-"+tgtlang, source_file), "r") as f:
        source_list = f.read().splitlines()
    with open(os.path.join(raw_dir, srclang+"-"+tgtlang, target_file), "r") as f:
        target_list = f.read().splitlines()
    
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

def load_local_dataset(cfg):
    with open(cfg["src_path"], "r") as f:
        source_list = f.read().splitlines()
    with open(cfg["tgt_path"], "r") as f:
        target_list = f.read().splitlines()
    
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
    elif source == "local":
        return load_local_dataset(cfg)
    else:
        raise NotImplementedError(f"Unknown source: {source}")
