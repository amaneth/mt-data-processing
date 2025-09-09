import requests

def opus_info(srclang, tgtlang):
    opus_url = (
        f"https://opus.nlpl.eu/opusapi/?source={srclang}&target={tgtlang}"
        "&preprocessing=moses&version=latest"
    )
    response = requests.get(opus_url)
    response_json = response.json()
    corpora = response_json["corpora"]

    print(f"\n🌍 Available corpora for {srclang} → {tgtlang}:\n")
    for entry in corpora:
        print(f"📘 Corpus : {entry['corpus']}")
        print(f"🔗 Pairs  : {entry.get('alignment_pairs', 'N/A')}")
        print(f"📦 Size   : {entry.get('size', 'N/A')} KB")
        print(f"📝 URL    : {entry['url']}\n")

    return corpora