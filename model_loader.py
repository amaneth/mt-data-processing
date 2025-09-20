from sentence_transformers import SentenceTransformer
from transformers import pipeline
from comet import download_model, load_from_checkpoint
import logging
import torch
import fasttext

logger = logging.getLogger("my_logger")

logging.getLogger("comet").propagate = False

def load_sentence_transformer(srclang, tgtlang):
    # Download and load the model
    model_cache = "model_cache"
    
    muse_langs = ['ar', 'de', 'en', 'es', 'fr', 'it', 'ko', 'nl', 'pt', 'pt', 'ru', 'tr', 'zh']
    para_langs = ["ar", "bg", "ca", "cs", "da", "de", "en", "el", "es", "et", "fa", "fi", "fr", "gl", "gu", "he", "hi", "hr", "hu", "hy", "id", "it", "ja", "ka", "ko", "ku", "lt", "lv", "mk", "mn", "mr", "ms", "my", "nb", "nl", "pl", "pt", "pt", "ro", "ru", "sk", "sl", "sq", "sr", "sv", "th", "tr", "uk", "ur", "vi", "zh"]
    microsoft_langs = ["en", "ar", "bg", "de", "el", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]
    labse_codes = [
    "af", "sq", "am", "ar", "hy", "as", "az", "eu", "be", "bn", "bs", "bg", "my", "ca", "ceb",
    "zh", "co", "hr", "cs", "da", "nl", "en", "eo", "et", "fi", "fr", "fy", "gl", "ka", "de",
    "el", "gu", "ht", "ha", "haw", "he", "hi", "hmn", "hu", "is", "ig", "id", "ga", "it", "ja",
    "jv", "kn", "kk", "km", "rw", "ko", "ku", "ky", "lo", "la", "lv", "lt", "lb", "mk", "mg",
    "ms", "ml", "mt", "mi", "mr", "mn", "ne", "no", "ny", "or", "fa", "pl", "pt", "pa", "ro",
    "ru", "sm", "gd", "sr", "st", "sn", "si", "sk", "sl", "so", "es", "su", "sw", "sv", "tl",
    "tg", "ta", "tt", "te", "th", "bo", "tr", "tk", "ug", "uk", "ur", "uz", "vi", "cy", "wo",
    "xh", "yi", "yo", "zu"
]

    
    if len(srclang) > 2 or len(tgtlang) > 2:
        raise SystemExit("Please use an ISO 639‑1 language code, e.g. 'en'!")
    elif srclang in muse_langs and tgtlang in muse_langs:
        model_name = "distiluse-base-multilingual-cased-v1"  # 15 languages
    elif srclang in para_langs and tgtlang in para_langs:
        model_name = "paraphrase-multilingual-MiniLM-L12-v2"  # 50 languages
    elif srclang in microsoft_langs and tgtlang in microsoft_langs:
        model_name = "microsoft/Multilingual-MiniLM-L12-H384"  # 16 language
    elif srclang in labse_codes and tgtlang in labse_codes:
        model_name = "sentence-transformers/LaBSE"
    else:
        raise SystemExit("Language pair is not supported!")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name, device=device, cache_folder=model_cache)
    logger.info(f"Loaded SentenceTransformer model: {model_name} on {device}")
    # pool = model.start_multi_process_pool()

    return model

def get_comet_model(model_name="masakhane/africomet-qe-stl"):
    model_path = download_model(model_name)  # downloads once and caches
    return load_from_checkpoint(model_path)

def get_fasttext_model(model_name="lid.176.bin"):
    model = fasttext.load_model("lid.176.bin")
    return model



def get_afrolid_model(model_name="UBC-NLP/afrolid_1.5"):
    model = pipeline("text-classification",
     model=model_name,
     device= torch.device("cuda" if torch.cuda.is_available() 
                          else "cpu")
     )
    return model



#HF - AfriDocMT-tech
