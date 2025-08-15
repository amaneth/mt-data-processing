import os
import yaml
from tabulate import tabulate
from datasets import Dataset
from dataset_loader import load_dataset_standard
from model_loader import load_sentence_transformer, get_comet_model

from pipelines import rule_filter, semantic_filter
from validators import language_detection, quality_estimation

import logging
from datetime import datetime
import sys
from itertools import chain
from tqdm import tqdm
import argparse
import json

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def save_to_file(lines, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def load_custom_metadata(output_dir):
    meta_path = os.path.join(output_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    return None

def setup_logging(debug, log_dir, log_file):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Prevent duplicate logs if this function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()
        
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Formatter for both console and file
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(message)s')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_log_path = os.path.join(log_dir, f"{timestamp}_{log_file}")
    file_handler = logging.FileHandler(full_log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logging initialized. Log file: {full_log_path}")

    return logger

def setup_logger(config):
    log_cfg = config["logging"]
    logger = setup_logging(
        debug=log_cfg.get("debug", True),
        log_dir=log_cfg.get("log_dir", "logs/"),
        log_file=log_cfg.get("log_file", "run.log")
    )
    return logger

def load_models(config):
    srclang, tgtlang = config["dataset"]["lang_pair"]
    sentence_model = load_sentence_transformer(srclang, tgtlang)
    model_pool = sentence_model.start_multi_process_pool()
    comet_model = get_comet_model(model_name="masakhane/africomet-qe-stl")
    return sentence_model, model_pool, comet_model

def collect_datasets(config):
    selected_sources = config["dataset"]["selected_sources"]
    all_datasets = list(chain.from_iterable(
        [dict(ds, source=source) for ds in config["dataset"].get(source, [])]
        for source in selected_sources
    ))
    raw_dir = config["download"]["output_dir"]
    os.makedirs(raw_dir, exist_ok=True)
    return all_datasets
def apply_rule_filter_if_enabled(source_list, target_list, config, logger):
    if config["preprocessing"].get("apply_rule_filter", True):
        rule_cfg = config["filters"]["rule_filter"]
        logger.info("🧹 Applying rule-based filtering...")
        source_list, target_list = rule_filter(
            source_texts=source_list,
            target_texts=target_list,
            min_length=rule_cfg.get("min_length", 3),
            max_length=rule_cfg.get("max_length", 200),
            max_length_ratio=rule_cfg.get("max_length_ratio", 2.0),
            lower=rule_cfg.get("lowercase", False),
        )
        logger.info(f"✅ Rule filter output: {len(source_list)} sentence pairs")
    return source_list, target_list

def apply_semantic_filter_if_enabled(source_list, target_list, config, srclang, tgtlang, logger, sentence_model, model_pool):
    if config["preprocessing"].get("apply_semantic_filter", False):
        sem_cfg = config["filters"]["semantic_filter"]
        logger.info("🧠 Applying semantic filtering...")
        source_list, target_list = semantic_filter(
            source_list,
            target_list,
            srclang=srclang,
            tgtlang=tgtlang,
            threshold=sem_cfg.get("threshold", 0.7),
            chunk_size=sem_cfg.get("chunk_size", 1000),
            batch_size=sem_cfg.get("batch_size", 2048),
            model=sentence_model,
            pool=model_pool
        )
        logger.info(f"✅ Semantic filter output: {len(source_list)} sentence pairs")
    return source_list, target_list

def run_validation(source_list, target_list, config, comet_model):
    lang_detected, lang_score, quality_score = None, None, None
    if config["validation"].get("language_detection", True):
        lang_detected, lang_score = language_detection(target_list)
    if config["validation"].get("quality_estimation", True):
        quality_score = quality_estimation(source_list, target_list, comet_model=comet_model)
    return lang_detected, lang_score, quality_score

def save_dataset(source_list, target_list, srclang, tgtlang, config, file_path, dataset_name, lang_pair, original_len, after_rule_len, after_semantic_len, lang_detected, lang_score, quality_score, logger):
    save_format = config["output"].get("save_format", "txt")
    if save_format == "hf":
        dataset_dict = {srclang: source_list, tgtlang: target_list}
        Dataset.from_dict(dataset_dict).save_to_disk(file_path)
        logger.info(f"✅ Hugging Face dataset saved to:\n  {file_path}")

        custom_metadata = {
            "dataset_name": dataset_name,
            "lang_pair": lang_pair,
            "original_rows": original_len,
            "after_rule": after_rule_len or original_len,
            "after_semantic": after_semantic_len or after_rule_len or original_len,
            "target_language": lang_detected,
            "lang_score": lang_score,
            "quality_score": quality_score,
            "processed_at": datetime.utcnow().isoformat()
        }
        with open(os.path.join(file_path, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(custom_metadata, f, ensure_ascii=False, indent=2)

    elif save_format == "txt":
        out_src = os.path.join(config["download"]["output_dir"], f"{config['output'].get('filtered_prefix', 'filtered')}.{srclang}")
        out_tgt = os.path.join(config["download"]["output_dir"], f"{config['output'].get('filtered_prefix', 'filtered')}.{tgtlang}")
        save_to_file(source_list, out_src)
        save_to_file(target_list, out_tgt)
        logger.info(f"✅ Saved filtered files:\n  - {out_src}\n  - {out_tgt}")

    else:
        raise ValueError(f"❌ Unknown output format: {save_format}")

def process_dataset(ds_cfg, config, logger, sentence_model, model_pool, comet_model):
    source = ds_cfg["source"]
    srclang, tgtlang = config["dataset"]["lang_pair"]
    output_prefix = config["output"].get("filtered_prefix", "filtered")
    raw_dir = config["download"]["output_dir"]
    name = ds_cfg['name']
    dataset_name = f"{output_prefix}-{name}"
    lang_pair = f"{srclang}-{tgtlang}"
    output_dir = config["output"].get("save_dir", os.path.join(raw_dir, "filtered_dataset"))
    file_path = os.path.join(output_dir, lang_pair, dataset_name)

    # Cache check
    if config["preprocessing"].get("cache", True) and os.path.exists(file_path):
        meta = load_custom_metadata(file_path)
        if meta:
            logger.info(f"✅ Cached: {meta['dataset_name']} — {meta['after_semantic']} pairs | QE: {meta['quality_score']}")
            return None
        logger.info(f"⚠ Found dataset at {output_dir} but no custom metadata — processing anyway")
        
    BLUE = "\033[1;34m"
    RESET = "\033[0m"
    logger.info(f"{BLUE}\n" + "=" * 80)
    logger.info(f"📦 STARTING DATASET: {source.upper()} - {name}")
    logger.info(f"🔤 Language Pair: {srclang}-{tgtlang}")
    logger.info(f"{"=" * 80}{RESET}")

    # Load data
    source_list, target_list = load_dataset_standard(ds_cfg, srclang, tgtlang, raw_dir=raw_dir, dataset_cache=config["download"].get("dataset_cache", "dataset_cache/"))
    if len(source_list) != len(target_list):
        logger.error(f"❌ Length mismatch. Source:{len(source_list)} Target:{len(target_list)}")
        sys.exit(1)

    original_len = len(source_list)
    source_list, target_list = apply_rule_filter_if_enabled(source_list, target_list, config, logger)
    after_rule_len = len(source_list)
    source_list, target_list = apply_semantic_filter_if_enabled(source_list, target_list, config, srclang, tgtlang, logger, sentence_model, model_pool)
    after_semantic_len = len(source_list)

    lang_detected, lang_score, quality_score = run_validation(source_list, target_list, config, comet_model)
    logger.info(f"✅ Validation done. Language:{lang_detected} Score:{lang_score} QE:{quality_score}")

    save_dataset(source_list, target_list, srclang, tgtlang, config, file_path, dataset_name, lang_pair, original_len, after_rule_len, after_semantic_len, lang_detected, lang_score, quality_score, logger)

    return {
        "source": ds_cfg["source"],
        "name": ds_cfg["name"],
        "original": original_len,
        "after_rule": after_rule_len,
        "after_semantic": after_semantic_len,
        "language_detected": (lang_detected, lang_score),
        "translation_quality": quality_score
    }

def log_final_summary(summary_log, logger):
    total_original = sum(entry["original"] for entry in summary_log)
    total_after_rule = sum(entry["after_rule"] for entry in summary_log)
    total_after_semantic = sum(entry["after_semantic"] for entry in summary_log)
    summary_table = [
        [entry["source"], entry["name"], entry["original"], entry["after_rule"], entry["after_semantic"], entry["language_detected"], entry["translation_quality"]]
        for entry in summary_log
    ]
    summary_table.append(["TOTAL", "-", total_original, total_after_rule, total_after_semantic, "-", "-"])
    logger.info("\n📊 Final Dataset Summary:\n" + tabulate(summary_table, headers=["Source", "Dataset", "Original", "After Rule", "After Semantic", "Language Detected", "Translation Quality"], tablefmt="github"))

def main(config_path):
    config = load_config(config_path)
    logger = setup_logger(config)
    logger.info("🚀 Starting preprocessing pipeline")
    sentence_model, model_pool, comet_model = load_models(config)
    summary_log = []
    datasets = collect_datasets(config)
    for ds_cfg in tqdm(datasets, desc="\033[1;34mProcessing datasets\033[0m"):
        summary = process_dataset(ds_cfg, config, logger, sentence_model, model_pool, comet_model)
        if summary:
            summary_log.append(summary)
    sentence_model.stop_multi_process_pool(model_pool)
    log_final_summary(summary_log, logger)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config YAML file(required)"
    )

    args = parser.parse_args()

    main(args.config)
