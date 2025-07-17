import requests
from wget import download
import requests
import shutil
import os
from tqdm.notebook import tqdm
import logging



def opus_info(srclang, tgtlang):
    opus_url = "https://opus.nlpl.eu/opusapi/?source="+srclang+"&target="+tgtlang+"&preprocessing=moses&version=latest"
    response = requests.get(opus_url)
    response_json = response.json()
    corpora = response_json["corpora"]
    
    logging.info(f"\nAvailable corpora for {srclang} → {tgtlang}:\n")
    for entry in corpora:
        logging.info(f"- Corpus: {entry['corpus']}")
        logging.info(f"  Pairs: {entry.get('alignment_pairs', 'N/A')}")
        logging.info(f"  Size: {entry.get('size', 'N/A')} KB")
        logging.info(f"  URL : {entry['url']}\n")

    return corpora




def download_opus(srclang, tgtlang, url, base_dir="raw/"):
    logging.info(f"Downloading opus data from {url}")
    filename = download(url)
    shutil.unpack_archive(filename, extract_dir=base_dir)
    os.remove(filename)

    directory = os.path.join(base_dir, srclang+"-"+tgtlang)
    os.makedirs(directory, exist_ok=True)

    unwanted_exts = (".ids", ".scores", ".xml", "LICENSE", "README")


    
    for filename in os.listdir(base_dir):
        path = os.path.join(base_dir, filename)

        if os.path.isdir(path):
            continue

        if filename.endswith(unwanted_exts):
            os.remove(path)
        elif filename.endswith(srclang):
            source_file = filename
            shutil.move(path, os.path.join(directory, filename))  
        elif filename.endswith(tgtlang):
            target_file = filename
            shutil.move(path, os.path.join(directory, filename)) 

    

    return source_file, target_file






                  
def download_url(src_url, tgt_url, src_name, tgt_name, base_dir="raw/"):

    src_text = requests.get(src_url)
    tgt_text = requests.get(tgt_url)
    src_path = os.path.join(base_dir, src_name)
    tgt_path = os.path.join(base_dir, tgt_name)

    with open(src_path, "w", encoding="utf-8") as f:
        f.write(src_text.text)
    with open(tgt_path, "w", encoding="utf-8") as f:
        f.write(tgt_text.text)

    logging.info(f"Download from url is done.\n Source file saved to: {src_path}\n target file saved to: {tgt_path}")


    return src_path, tgt_path


    




if __name__ == "__main__":
    # src_url = "https://raw.githubusercontent.com/AAUThematic4LT/Parallel-Corpora-for-Ethiopian-Languages/refs/heads/master/Exp%20I-English%20to%20Local%20Lang/History/amh_eng/p_eng_ea"
    # tgt_url = "https://raw.githubusercontent.com/AAUThematic4LT/Parallel-Corpora-for-Ethiopian-Languages/refs/heads/master/Exp%20I-English%20to%20Local%20Lang/History/amh_eng/p_amh_ea"
    # download_url(src_url, tgt_url, "eng.txt", "amh.txt")

    # download_opus("en", "am", ["wikimedia"])
    # opus_info("en", "am")
    url = "https://object.pouta.csc.fi/OPUS-wikimedia/v20230407/moses/am-en.txt.zip"
    download_opus_single("en", "am", url)