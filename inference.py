import argparse

from src import models
from utils import *

text = "28-Year-Old Chef Found Dead at San Francisco Mall"

def main(args):
    config = load_config("cof/information_extraction_base_bert.yml")
    """ get net struction"""
    net = models[config["model_type"]](**config["model"])
    net.load_pretrained(config["logging"]["save_dir"])
    translate = net.infernece(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path",type=str,default = "config/entity_extraction_base_bert.yml")
    args = parser.parse_args()
    main(args)

