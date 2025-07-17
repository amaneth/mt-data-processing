import os
import yaml
from datasets import load_dataset, Dataset

from pipelines import rule_filter, semantic_filter
from fetch import download_url, download_opus

import logging
from datetime import datetime
import argparse

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def save_to_file(lines, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def setup_logging(debug, log_dir, log_file):
    os.makedirs(log_dir, exist_ok=True)

    log_level = logging.DEBUG if debug else logging.INFO
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_log_path = os.path.join(log_dir, f"{timestamp}_{log_file}")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s", 
        handlers=[
            logging.FileHandler(full_log_path),
            logging.StreamHandler()
        ]

    )

    logging.info(f"Logging initialized. Log file: {full_log_path}")


def main():
    config_path = "configs/am_config.yaml"
    config = load_config(config_path)

    # Setup logging
    log_cfg = config.get("logging", {})
    setup_logging(
        debug=log_cfg.get("debug", False),
        log_dir=log_cfg.get("log_dir", "log"),
        log_file=log_cfg.get("log_file", "pipeline.log")
    ) 
    logging.info("🚀 Starting preprocessing pipeline")
    

    selected_sources = config["dataset"]["selected_sources"]
    lang_pair = config["dataset"]["lang_pair"]
    srclang, tgtlang = lang_pair
    raw_dir = config["download"]["output_dir"]
    os.makedirs(raw_dir, exist_ok=True)
    for source in selected_sources:
        logging.info(f"Dataset source: {source.upper()} | Language pair: {srclang}-{tgtlang}")

        datasets = config["dataset"].get(source, [])
        for ds in datasets:
            name = ds["name"]
            logging.info(f"🚀 Processing {source.upper()} dataset: {name}")

            if source == "hf":
                if "config_name" in ds:
                    dataset = load_dataset(ds["path"], name=ds["config_name"], split=ds["split"])
                else:
                    dataset = load_dataset(ds["path"], split=ds["split"])

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
                source_file, target_file = download_opus(srclang, tgtlang, url, base_dir=raw_dir)
                with open(os.path.join(raw_dir, source_file), "r") as f:
                    source_list = f.read().splitlines()
                with open(os.path.join(raw_dir, target_file), "r") as f:
                    target_list = f.read().splitlines()


            # Step 2: Rule filtering
            if config["preprocessing"].get("apply_rule_filter", True):
                rule_cfg = config["filters"]["rule_filter"]
                logging.info("🧹 Applying rule-based filtering...")

                source_list, target_list = rule_filter(
                    source_texts=source_list,
                    target_texts=target_list,
                    min_length=rule_cfg.get("min_length", 3),
                    max_length=rule_cfg.get("max_length", 200),
                    max_length_ratio=rule_cfg.get("max_length_ratio", 2.0),
                    lower=rule_cfg.get("lowercase", False),
                )
                logging.info(f"✅ Rule filter output: {len(source_list)} sentence pairs")


            # Step 3: Semantic filtering
            if config["preprocessing"].get("apply_semantic_filter", False):
                sem_cfg = config["filters"]["semantic_filter"]
                logging.info("🧠 Applying semantic filtering...")
                logging.debug(f"Semantic filter config: {sem_cfg}")
                source_list, target_list = semantic_filter(
                    source_list,
                    target_list,
                    srclang=srclang,
                    tgtlang=tgtlang,
                    threshold=sem_cfg.get("threshold", 0.7),
                    chunk_size=sem_cfg.get("chunk_size", 1000),
                    batch_size=sem_cfg.get("batch_size", 2048)
                )
                logging.info(f"✅ Semantic filter output: {len(source_list)} sentence pairs")

            # Step 4: Save final outputs
            output_prefix = config["output"].get("filtered_prefix", "filtered")
            save_format = config["output"].get("save_format", "txt")

            logging.info("💾 Saving final dataset...")
            if save_format=="hf":
                dataset_dict = {
                    srclang: source_list,
                    tgtlang: target_list,
                }

                final_dataset = Dataset.from_dict(dataset_dict)

                output_dir = config["output"].get("save_dir", os.path.join(raw_dir, "filtered_dataset"))
                final_dataset.save_to_disk(output_dir)

                logging.info(f"✅ Hugging Face dataset saved to:\n  {output_dir}")
            elif save_format=="txt":
                out_src = os.path.join(raw_dir, f"{output_prefix}.{srclang}")
                out_tgt = os.path.join(raw_dir, f"{output_prefix}.{tgtlang}")

                save_to_file(source_list, out_src)
                save_to_file(target_list, out_tgt)

                logging.info(f"✅ Done! Saved filtered files to:\n  - {out_src}\n  - {out_tgt}")

            else:
                logging.error(f"❌ Unknown output format: {save_format}")
                raise ValueError(f"Unknown output format: {save_format}")




if __name__ == "__main__":
    main()
