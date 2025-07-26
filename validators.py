import fasttext
import argparse
from comet import download_model, load_from_checkpoint

import logging

logger = logging.getLogger("my_logger")

logging.getLogger("comet").propagate = False

def load_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().splitlines()

def get_comet_model(model_name="masakhane/africomet-qe-stl"):
    model_path = download_model(model_name)  # downloads once and caches
    return load_from_checkpoint(model_path)


# def language_detection(target_list):

#     model = fasttext.load_model("lid.176.bin")

#     content = " ".join(target_list)

#     label, confidence = model.predict(content)

#     predicted_langauge = label[0].replace('__label__', '')
#     confidence = confidence[0]

#     logger.info(f"🧠 Language Detection → Language: {predicted_langauge.upper()} | Confidence: {confidence[0]:.4f}")

#     return predicted_langauge, confidence


def language_detection(texts, expected_lang=None, threshold=0.6):
    model = fasttext.load_model("lid.176.bin")

    lang_counts = {}
    total = 0

    for line in texts:
        line = line.strip().replace("\n", " ")
        if not line:
            continue

        label, confidence = model.predict(line)
        lang = label[0].replace("__label__", "")
        conf = confidence[0]

        # Optional: ignore low-confidence lines
        if conf < threshold:
            continue

        lang_counts[lang] = lang_counts.get(lang, 0) + 1
        total += 1

    if not total:
        logger.warning("⚠️ No confident language predictions found.")
        return None, 0.0

    # Get the most common predicted language
    predicted_lang = max(lang_counts, key=lang_counts.get)
    purity_score = lang_counts[predicted_lang] / total

    if expected_lang:
        status = "✅" if predicted_lang == expected_lang else "⚠️"
        logger.info(
            f"{status} Language Detection → Detected: {predicted_lang.upper()} | Confidence: {purity_score:.4f} | Expected: {expected_lang.upper()}"
        )
    else:
        logger.info(
            f"🧠 Language Detection → Dominant: {predicted_lang.upper()} | Confidence: {purity_score:.4f}"
        )

    return predicted_lang, purity_score






def quality_estimation(source_list, target_list):
    model_name="masakhane/africomet-qe-stl"
    model = get_comet_model(model_name)
    
    data = [{"src": src.strip(), "mt": tgt.strip()} for src, tgt in zip(source_list, target_list)]

    # Predict
    model_output = model.predict(data, batch_size=8, gpus=1)
    system_score = round(model_output.system_score, 4)

    logger.info(f"🧪 Quality Estimation → Score:{system_score}")
    return system_score


