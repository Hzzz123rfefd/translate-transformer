from src.model import ModelForTranslate,ModelPretrainForTranslateBaseBart
from src.dataset import DatasetForTranslate


datasets = {
   "translate": DatasetForTranslate
}

models = {
    "translate_from_pretrained": ModelPretrainForTranslateBaseBart,
    "translate": ModelForTranslate
}