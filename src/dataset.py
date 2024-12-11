
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

import datasets
from torch.utils.data import Dataset
from src.utils import *


class DatasetForTranslate(Dataset):
    def __init__(
            self,
            train_data_path:str = None,
            test_data_path:str = None,
            valid_data_path:str = None,
            max_padding_length:int = 512,
            tokenizer: PreTrainedTokenizer = None,
            data_type:str = "train"
    ):
        # load data
        if data_type == "train":
            self.dataset = datasets.load_dataset('json', data_files = train_data_path,split = "train")
        elif data_type == "test":
            self.dataset = datasets.load_dataset('json', data_files = test_data_path,split = "train")
        elif data_type == "valid":
            self.dataset = datasets.load_dataset('json', data_files = valid_data_path,split = "train")

        self.total_len = len(self.dataset)
        self.max_padding_length = max_padding_length
        self.tokenizer = tokenizer
    
    def encode_text(self, text:str):
        items = self.tokenizer(
            text, 
            return_tensors = "pt", 
            truncation = True, 
            max_length = self.max_padding_length
        )
        return items["input_ids"],items["attention_mask"]

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        x_encoder = {}
        x_decoder = {}
        output = {
            "x_encoder":x_encoder,
            "x_decoder":x_decoder,
        }
        input_ids, attention_mask = self.encode_text(self.dataset[idx]['text'])
        output["x_encoder"]["input_ids"] = input_ids
        output["x_encoder"]["attention_mask"] = attention_mask

        input_ids, attention_mask = self.encode_text(self.dataset[idx]['label'])
        output["x_decoder"]["input_ids"] = input_ids
        output["x_decoder"]["attention_mask"] = attention_mask

        return output
    
    def collate_fn(self,batch):
        max_length_encoder = max(item["x_encoder"]["input_ids"].shape[1] for item in batch)
        max_length_decoder = max(item["x_decoder"]["input_ids"].shape[1] for item in batch)
        
        for item in batch:
            item["x_encoder"] = pad_sequence(item['x_encoder'],max_length_encoder)
            item["x_decoder"] = pad_sequence(item['x_decoder'],max_length_decoder)

        return  recursive_collate_fn(batch)