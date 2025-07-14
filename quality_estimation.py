import fasttext
import argparse
from comet import download_model, load_from_checkpoint


def load_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().splitlines()

def get_comet_model(model_name="masakhane/africomet-qe-stl"):
    model_path = download_model(model_name)  # downloads once and caches
    return load_from_checkpoint(model_path)


def language_detection(target_list):

    model = fasttext.load_model("../lid.176.bin")

    content = " ".join(target_list)

    label, confidence = model.predict(content)

    predicted_langauge = label[0].replace('__label__', '')
    confidence = confidence[0]

    print(f"Predicted Language: {predicted_langauge}")
    print(f"Confidence: {confidence[0]:.4f}")


    return predicted_langauge, confidence





def quality_estimation(source_list, target_list):
    model_name="masakhane/africomet-qe-stl"
    model = get_comet_model(model_name)
    
    data = [{"src": src.strip(), "mt": tgt.strip()} for src, tgt in zip(source_list, target_list)]

    # Predict
    model_output = model.predict(data, batch_size=8, gpus=1)



    system_score = round(model_output.system_score, 4)
    # Display results
    print(f"System Score: {system_score}")
    return system_score


def main():
    parser = argparse.ArgumentParser(description="Run language detection or QE on parallel data")
    parser.add_argument("--task", required=True, choices=["lang", "qe"], help="Task to run: 'lang' or 'qe'")
    parser.add_argument("--source", required=False, help="Path to source file")
    parser.add_argument("--target", required=True, help="Path to target file")
    args = parser.parse_args()

    target_list = load_file(args.target)

    if args.task == "lang":
        language_detection(target_list)

    elif args.task == "qe":
        if not args.source:
            raise ValueError("For quality estimation (--task qe), you must provide --source")
        source_list = load_file(args.source)
        quality_estimation(source_list, target_list)

if __name__ == "__main__":
    main()