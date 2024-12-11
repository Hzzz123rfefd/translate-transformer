from torch.utils.data.dataloader import default_collate
import torch

def recursive_collate_fn(batch):
    if isinstance(batch[0], dict):
        return {key: recursive_collate_fn([b[key] for b in batch]) for key in batch[0]}
    else:
        return default_collate(batch)
    
def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def pad_sequence(
    sequence:dict, 
    max_length:int,
    pad_token_id:int = 0
) -> dict:
    
    seq_length = sequence['input_ids'].shape[1]
    padded_sequence = torch.ones((sequence['input_ids'].shape[0], max_length), dtype = sequence['input_ids'].dtype) * pad_token_id
    attention_mask = torch.ones((sequence['input_ids'].shape[0], max_length), dtype = sequence['attention_mask'].dtype) * pad_token_id

    padded_sequence[:, :seq_length] = sequence['input_ids']
    attention_mask[:, :seq_length] = sequence['attention_mask']

    sequence["input_ids"] = padded_sequence
    sequence["attention_mask"] = attention_mask

    return sequence