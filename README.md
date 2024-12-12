# translate-transformer
Machine Translation Using Transformer Architecture, Including custom transformer architecture and pre trained bart architecture


## Installation
Install the packages required for development.
```bash
conda create -n translate python=3.10
conda activate translate
git clone https://github.com/Hzzz123rfefd/translate-transformer.git
cd translate-transformer
pip install -r requirements.txt
```

## Usage
### Dataset
Firstly, you can download the wmt zh to en dataset,you can download it  with following script:
```bash
python datasets/wmt17/download.py --output_dir wmt17_train/
```
your directory structure should be:
- translate-transformer/
  - wmt17_train/
    - train.jsonl
    - test.jsonl
    - vaild.jsonl

No matter what dataset you use, please convert it to the required dataset format for this project, as follows (you can also view it in `data/train.jsonl`)
```json
{"text":"text1","label":"文本1"}
```

### Trainning
* An examplary training script is provided in `train.py`.
* We have prepared two models here, one is the native transformer model and the other is the pre trained bart model
* You can adjust the model parameters in `config/translate.yml` and `config/translate_base_bart.yml`, there are detailed parameter descriptions there, you can modify them according to your needs
```bash
python train.py --model_config_path config/translate_base_bart.yml
```

### Inference
Once you have trained your model, you can use the following script to perform translate on the data
you can set your text in `inference.py`
```bash
python inference.py --model_config_path config/translate_base_bart.yml
```
TODO:



