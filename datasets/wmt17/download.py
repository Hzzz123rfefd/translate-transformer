import argparse
from datasets import load_dataset
import json
import sys
import os
sys.path.append(os.getcwd())

def main(args):
    dataset = load_dataset("wmt17", "zh-en")

    for split in dataset.keys():
        output_file = args.output_dir + f"wmt_en_zh_{split}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for example in dataset[split]:
                text = example['translation']['en']  
                label = example['translation']['zh']  
                json_line = json.dumps({"text": text, "label": label}, ensure_ascii=False)
                f.write(json_line + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args("--output_dir",type=str,default = "wmt17_train/")
    main(args)
