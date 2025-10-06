import argparse
import yaml
import random
from datasets import load_dataset, Dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Merge and sample NLLB datasets.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--output", type=str, required=True, help="Output path for merged HF dataset.")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def process_dataset(cfg):
    ds = load_dataset(cfg["path"], split=cfg["split"])
    src_col = cfg["src_col"]
    tgt_col = cfg["tgt_col"]
    src_lang = cfg["src_lang"]
    tgt_lang = cfg["tgt_lang"]
    sample_size = cfg["sample_size"]

    # If sample_size is float (fraction), convert to int
    if isinstance(sample_size, float) and 0 < sample_size <= 1:
        sample_size = int(sample_size * len(ds))
    elif isinstance(sample_size, int):
        sample_size = min(sample_size, len(ds))
    else:
        sample_size = len(ds)

    indices = random.sample(range(len(ds)), sample_size)
    sampled = ds.select(indices)

    rows = []
    for item in sampled:
        rows.append({
            f"sentence_{src_lang}": item[src_col],
            f"sentence_{tgt_lang}": item[tgt_col],
            "src_lang": src_lang,
            "tgt_lang": tgt_lang
        })
    return rows

def main():
    args = parse_args()
    config = load_config(args.config)

    all_rows = []
    for ds_cfg in config.values():
        all_rows.extend(process_dataset(ds_cfg))

    merged_dataset = Dataset.from_list(all_rows)
    merged_dataset.save_to_disk(args.output)
    print(f"Merged dataset saved to {args.output}")

if __name__ == "__main__":
    main()