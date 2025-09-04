#!/usr/bin/env python3
import os
import yaml
from pathlib import Path

# Mapping from config file to language code
LANG_MAPPING = {
    'af_config.yaml': 'af',
    'am_config.yaml': 'am',
    'hu_config.yaml': 'hu',
    'so_config.yaml': 'so',
    'sw_config.yaml': 'sw',
    'zu_config.yaml': 'zu'
}

# URL base pattern for OPUS datasets
OPUS_BASE_URL = "https://object.pouta.csc.fi/OPUS-{corpus}/v{version}/moses/{src}-{tgt}.txt.zip"

def update_config_file(config_path, tgt_lang):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update lang_pair
    config['dataset']['lang_pair'] = ["en", tgt_lang]

    # Update config names for Weblate and Pontoon
    if 'hf' in config['dataset']:
        for item in config['dataset']['hf']:
            if item.get('config_name') == 'en-am':
                item['config_name'] = f'en-{tgt_lang}'
            if item.get('path') == 'ayymen/Pontoon-Translations' and item.get('config_name'):
                item['config_name'] = f'en-{tgt_lang}'
            if item.get('path') == 'ayymen/Weblate-Translations' and item.get('config_name'):
                item['config_name'] = f'en-{tgt_lang}'

    # Remove URL section for non-Amharic configs (am_config.yaml has specific URLs we want to keep)
    if tgt_lang != 'am' and 'url' in config['dataset']:
        del config['dataset']['url']

    # Update table section for non-Amharic configs
    if tgt_lang != 'am' and 'table' in config['dataset']:
        del config['dataset']['table']

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

if __name__ == "__main__":
    configs_dir = Path("configs")

    for config_file, tgt_lang in LANG_MAPPING.items():
        config_path = configs_dir / config_file
        if config_path.exists():
            print(f"Updating {config_path}...")
            update_config_file(config_path, tgt_lang)
            print(f"✅ Updated lang_pair to ['en', '{tgt_lang}']")

    print("All config files updated!")