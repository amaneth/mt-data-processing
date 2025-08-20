import fasttext
import argparse


import logging

logger = logging.getLogger("my_logger")


def load_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().splitlines()





def quality_estimation(source_list, target_list, comet_model=None):
    
    data = [{"src": src.strip(), "mt": tgt.strip()} for src, tgt in zip(source_list, target_list)]

    # Predict
    model_output = comet_model.predict(data, batch_size=8, gpus=1)
    system_score = round(model_output.system_score, 4)

    logger.info(f"🧪 Quality Estimation → Score:{system_score}")
    return system_score


