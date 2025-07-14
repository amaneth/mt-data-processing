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
    
    logging.debug(f"\nAvailable corpora for {srclang} → {tgtlang}:\n")
    for entry in corpora:
        logging.debug(f"- Corpus: {entry['corpus']}")
        logging.debug(f"  Pairs: {entry.get('alignment_pairs', 'N/A')}")
        logging.debug(f"  Size: {entry.get('size', 'N/A')} KB")
        logging.debug(f"  URL : {entry['url']}\n")

    return corpora

def download_opus(srclang, tgtlang, corpora_names, base_dir="raw/"):

    corpora = opus_info(srclang, tgtlang)
    # Get data size

    target_corpora = [corpus for corpus in corpora if corpus["corpus"] in corpora_names]

    data_size = 0
    for corpus in target_corpora:
        if corpus["alignment_pairs"]:
            data_size += corpus["alignment_pairs"]

        logging.info("Line count:", format(data_size, ','))
    
    # print(f"The current working dir:{os.getcwd()}: files in cwd: {os.listdir()}")
    # os.chdir(base_dir)
    # print(f"The current working dir:{os.getcwd()}: files in cwd: {os.listdir()}")
    for corpus in target_corpora:
        logging.info("•", corpus["corpus"])
        filename = download(corpus["url"])
        shutil.unpack_archive(filename, extract_dir=base_dir)
        os.remove(filename)


    directory = os.pteaath.join(base_dir, srclang+"-"+tgtlang)
    os.makedirs(directory, exist_ok=True)

    unwanted_exts = (".ids", ".scores", ".xml", "LICENSE", "README")
    
    source_files, target_files = [],[]
    for filename in os.listdir(base_dir):
        path = os.path.join(base_dir, filename)

        if os.path.isdir(path):
            continue

        if filename.endswith(unwanted_exts):
            os.remove(path)
        elif filename.endswith(srclang):
            source_files.append(filename)
            shutil.move(path, os.path.join(directory, filename))  
        elif filename.endswith(tgtlang):
            target_files.append(filename)
            shutil.move(path, os.path.join(directory, filename))   

    if len(source_files) == len(target_files):
        logging.info("Saved source files:", *sorted(source_files))
        logging.info("Saved target files:", *sorted(target_files))
    else:
        logging.info(f"Different number of source and target files: {len(source_files)} source files vs {len(target_files)} target files")


    return source_files, target_files

                  
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

    download_opus("en", "am", ["wikimedia"])