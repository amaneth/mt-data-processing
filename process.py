import os
import yaml
from tabulate import tabulate
from datasets import load_dataset, Dataset

from pipelines import rule_filter, semantic_filter
from fetch import download_url, download_opus
from validators import language_detection, quality_estimation

import logging
from datetime import datetime
import argparse
import sys
from itertools import chain
from tqdm import tqdm

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def save_to_file(lines, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

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


def main():
    dataset_cache = "dataset_cache"
    config_path = "configs/am_config.yaml"
    config = load_config(config_path)

    # Setup logging
    log_cfg = config["logging"]
    logger = setup_logging(
        debug=log_cfg.get("debug", True), 
        log_dir=log_cfg.get("log_dir", "logs/"),
        log_file=log_cfg.get("log_file", "run.log"))
    logger.info("🚀 Starting preprocessing pipeline")
    
    summary_log = []
    
    selected_sources = config["dataset"]["selected_sources"]
    lang_pair = config["dataset"]["lang_pair"]
    srclang, tgtlang = lang_pair
    raw_dir = config["download"]["output_dir"]
    os.makedirs(raw_dir, exist_ok=True)

    all_datasets = list(chain.from_iterable(
        [(source, ds) for ds in config["dataset"].get(source, [])]
        for source in selected_sources
    ))
    BLUE = "\033[1;34m"
    RESET = "\033[0m"
    for dataset_counter, (source, ds) in enumerate(tqdm(all_datasets, desc=f"{BLUE}Processing datasets"), 1):
        name = ds["name"]

        logger.info(f"{BLUE}\n" + "=" * 80)
        logger.info(f"📦 STARTING DATASET: {source.upper()} - {name}")
        logger.info(f"🔤 Language Pair: {srclang}-{tgtlang}")
        logger.info(f"{"=" * 80}{RESET}")

        if source == "hf":
            if "config_name" in ds:
                dataset = load_dataset(ds["path"], name=ds["config_name"], split=ds["split"], cache_dir=dataset_cache)
            else:
                dataset = load_dataset(ds["path"], split=ds["split"], cache_dir=dataset_cache)

            source_list = [item[ds["src_col"]] for item in dataset]
            target_list = [item[ds["tgt_col"]] for item in dataset]

        elif source == "github":
            download_url(
                ds["src_url"],
                ds["tgt_url"],
                src_name=f"{name}.{srclang}",
                tgt_name=f"{name}.{tgtlang}",
                base_dir=raw_dir,
            )
            with open(os.path.join(raw_dir, f"{name}.{srclang}"), "r") as f:
                source_list = f.read().splitlines()
            with open(os.path.join(raw_dir, f"{name}.{tgtlang}"), "r") as f:
                target_list = f.read().splitlines()

        elif source == "opus":
            url = ds["url"]
            name = ds["name"]
            source_file, target_file = download_opus(srclang, tgtlang, name, url, base_dir=raw_dir)
            with open(os.path.join(raw_dir, srclang+"-"+tgtlang, source_file), "r") as f:
                source_list = f.read().splitlines()
            with open(os.path.join(raw_dir, srclang+"-"+tgtlang, target_file), "r") as f:
                target_list = f.read().splitlines()

            
        if len(source_list) != len(target_list):
            logger.debug(f"❌ Length mismatch. Source:{len(source_list)} Target:{len(target_list)}")
            sys.exit(1)

        original_len = len(source_list)
        after_rule_len = None
        after_semantic_len = None
        # Step 2: Rule filtering
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
            after_rule_len = len(source_list)
            logger.info(f"✅ Rule filter output: {len(source_list)} sentence pairs")


        # Step 3: Semantic filtering
        if config["preprocessing"].get("apply_semantic_filter", False):
            sem_cfg = config["filters"]["semantic_filter"]
            logger.info("🧠 Applying semantic filtering...")
            logger.debug(f"Semantic filter config: {sem_cfg}")
            source_list, target_list = semantic_filter(
                source_list,
                target_list,
                srclang=srclang,
                tgtlang=tgtlang,
                threshold=sem_cfg.get("threshold", 0.7),
                chunk_size=sem_cfg.get("chunk_size", 1000),
                batch_size=sem_cfg.get("batch_size", 2048)
            )
            after_semantic_len = len(source_list)
            logger.info(f"✅ Semantic filter output: {len(source_list)} sentence pairs")


        # Step 4: Validation 
        if config["validation"].get("language_detection", True): 
            lang_detected, lang_score = language_detection(target_list)
        if config["validation"].get("quality_estimation", True):
            quality_score = quality_estimation(source_list, target_list)
        
        logger.info(f"✅ Validation done. Language:{lang_detected} Language Score: {lang_score} QE socre: {quality_score}")

        summary_log.append({
            "source": source,
            "name": name,
            "original": original_len,
            "after_rule": after_rule_len or original_len,
            "after_semantic": after_semantic_len or after_rule_len or original_len,
            "language_detected": (lang_detected, lang_score),
            "translation_quality": quality_score
        })



    
        # Step 5: Save final outputs
        output_prefix = config["output"].get("filtered_prefix", "filtered")
        save_format = config["output"].get("save_format", "txt")

        logger.info("💾 Saving final dataset...")
        if save_format=="hf":
            dataset_dict = {
                srclang: source_list,
                tgtlang: target_list,
            }

            final_dataset = Dataset.from_dict(dataset_dict)


            output_dir = config["output"].get("save_dir", os.path.join(raw_dir, "filtered_dataset"))
            file_path = os.path.join(output_dir, srclang+"-"+tgtlang, output_prefix+"-"+name)
            final_dataset.save_to_disk(file_path)

            logger.info(f"✅ Hugging Face dataset saved to:\n  {file_path}")
        elif save_format=="txt":
            out_src = os.path.join(raw_dir, f"{output_prefix}.{srclang}")
            out_tgt = os.path.join(raw_dir, f"{output_prefix}.{tgtlang}")

            save_to_file(source_list, out_src)
            save_to_file(target_list, out_tgt)

            logger.info(f"✅ Done! Saved filtered files to:\n  - {out_src}\n  - {out_tgt}")

        else:
            logging.error(f"❌ Unknown output format: {save_format}")
            raise ValueError(f"Unknown output format: {save_format}")
    logger.info("\n📊 Final Dataset Summary:")
    logger.info(
        "\n" + tabulate(
            [[entry["source"], entry["name"], entry["original"], entry["after_rule"], entry["after_semantic"], entry["language_detected"], entry["translation_quality"]]
            for entry in summary_log],
            headers=["Source", "Dataset", "Original", "After Rule", "After Semantic", "Language Detected", "Translation Quality"],
            tablefmt="github"
        )
    )




if __name__ == "__main__":
    main()
