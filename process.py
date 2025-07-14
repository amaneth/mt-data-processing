import os
import yaml
from datasets import load_dataset, Dataset

from pipelines import rule_filter, semantic_filter
from fetch import download_url, download_opus

import logging
from datetime import datetime

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

    dataset_cfg = config["dataset"]
    source = dataset_cfg["source"]
    lang_pair = dataset_cfg["lang_pair"]
    srclang, tgtlang = lang_pair
    raw_dir = config["download"]["output_dir"]
    
    # Setup logging
    log_cfg = config.get("logging", {})
    setup_logging(
        debug=log_cfg.get("debug", False),
        log_dir=log_cfg.get("log_dir", "log"),
        log_file=log_cfg.get("log_file", "pipeline.log")
    ) 
    logging.info("🚀 Starting preprocessing pipeline")
    logging.info(f"Dataset source: {source.upper()} | Language pair: {srclang}-{tgtlang}")
    os.makedirs(raw_dir, exist_ok=True)

    # Step 1: Download data
    if source == "hf":
        hf = dataset_cfg["hf"]
        logging.debug(f"Loading from HF dataset: {hf['path']}, split: {hf['split']}")
        dataset = load_dataset(hf["path"], split=hf["split"])
        source_list = [item[hf["src_lang"]] for item in dataset["translation"]]
        target_list = [item[hf["tgt_lang"]] for item in dataset["translation"]]

    elif source == "github":
        github = dataset_cfg["github"]
        src_path = os.path.join(raw_dir, f"raw.{srclang}")
        tgt_path = os.path.join(raw_dir, f"raw.{tgtlang}")
        logging.debug(f"GitHub URLs: {github['src_url']} | {github['tgt_url']}")
        logging.debug(f"Download paths: {src_path}, {tgt_path}")


        download_url(
            github["src_url"],
            github["tgt_url"],
            src_name=f"raw.{srclang}",
            tgt_name=f"raw.{tgtlang}",
            base_dir=raw_dir,
        )

        with open(src_path, "r", encoding="utf-8") as f:
            source_list = f.read().splitlines()
        with open(tgt_path, "r", encoding="utf-8") as f:
            target_list = f.read().splitlines()

    elif source == "opus":
        logging.debug(f"Downloading OPUS corpora: {dataset_cfg['opus']['corpora_names']}")
        download_opus(srclang, tgtlang, dataset_cfg["opus"]["corpora_names"], base_dir=raw_dir)
        src_path = os.path.join(raw_dir, f"raw.{srclang}")
        tgt_path = os.path.join(raw_dir, f"raw.{tgtlang}")

        with open(src_path, "r", encoding="utf-8") as f:
            source_list = f.read().splitlines()
        with open(tgt_path, "r", encoding="utf-8") as f:
            target_list = f.read().splitlines()

    else:
        logging.error(f"Unknown dataset source: {source}")
        raise ValueError(f"Unknown dataset source: {source}")

    logging.info(f"📄 Loaded {len(source_list)} sentence pairs")

    if len(source_list) != len(target_list):
        logging.warning("⚠️ Source and target list lengths are not equal!")


    # Step 2: Rule filtering
    if config["preprocessing"].get("apply_rule_filter", True):
        lower = config["preprocessing"].get("lowercase", False)
        logging.info("🧹 Applying rule-based filtering...")
        logging.debug(f"Lowercasing: {lower}")

        source_list, target_list = rule_filter(
            source_texts=source_list,
            target_texts=target_list,
            source_lang=srclang,
            target_lang=tgtlang,
            lower=lower,
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
