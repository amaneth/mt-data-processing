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
import argparse
import sys
from itertools import chain
from tqdm import tqdm
import argparse

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


def main(config_path):
    # config_path = "configs/am_config.yaml"
    config = load_config(config_path)
    dataset_cache = config["download"].get("dataset_cache", "dataset_cache/")

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
        [dict(ds, source=source) for ds in config["dataset"].get(source, [])]
        for source in selected_sources
    ))

    # load sentence_transformers and comet models
    sentence_transformer_model = load_sentence_transformer(srclang, tgtlang)
    model_pool=sentence_transformer_model.start_multi_process_pool()
    comet_model = get_comet_model(model_name="masakhane/africomet-qe-stl")

    BLUE = "\033[1;34m"
    RESET = "\033[0m"
    for ds_cfg in tqdm(all_datasets, desc=f"{BLUE}Processing datasets"):
        name = ds_cfg["name"]
        source = ds_cfg["source"]
        srclang, tgtlang = config["dataset"].get("lang_pair")

        logger.info(f"{BLUE}\n" + "=" * 80)
        logger.info(f"📦 STARTING DATASET: {source.upper()} - {name}")
        logger.info(f"🔤 Language Pair: {srclang}-{tgtlang}")
        logger.info(f"{"=" * 80}{RESET}")

        source_list, target_list = load_dataset_standard(ds_cfg, srclang, tgtlang, raw_dir=raw_dir, dataset_cache=dataset_cache)
        
            
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
                batch_size=sem_cfg.get("batch_size", 2048),
                model=sentence_transformer_model,
                pool=model_pool
            )
         
            
            after_semantic_len = len(source_list)
            logger.info(f"✅ Semantic filter output: {len(source_list)} sentence pairs")


        # Step 4: Validation 
        if config["validation"].get("language_detection", True): 
            lang_detected, lang_score = language_detection(target_list)
        if config["validation"].get("quality_estimation", True):
            quality_score = quality_estimation(source_list, target_list, comet_model=comet_model)
        
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

        # Compute total counts
    sentence_transformer_model.stop_multi_process_pool(model_pool)
    total_original = sum(entry["original"] for entry in summary_log)
    total_after_rule = sum(entry["after_rule"] for entry in summary_log)
    total_after_semantic = sum(entry["after_semantic"] for entry in summary_log)

        # Add a summary row
    summary_table = [
        [entry["source"], entry["name"], entry["original"], entry["after_rule"], entry["after_semantic"], entry["language_detected"], entry["translation_quality"]]
        for entry in summary_log
    ]

    # Append total row
    summary_table.append([
        "TOTAL", "-", total_original, total_after_rule, total_after_semantic, "-", "-"
    ])


    logger.info("\n📊 Final Dataset Summary:")
    logger.info(
        "\n" + tabulate(
            summary_table,
            headers=["Source", "Dataset", "Original", "After Rule", "After Semantic", "Language Detected", "Translation Quality"],
            tablefmt="github"
        )
    )




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
