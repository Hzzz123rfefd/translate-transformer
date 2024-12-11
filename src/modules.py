import math
import torch
from torch import nn

class ModelTransformer(nn.Module):
    def __init__(self, 
        d_model_encoder, 
        d_model_decoder, 
        n_head,
        ffn_hidden, 
        n_layers, 
        drop_prob = 0.1, 
        max_seq_len = 1024,
        device = "cpu"
    ):
        super().__init__()
        self.device = device
        self.max_seq_len = max_seq_len
        self.d_model_decoder = d_model_decoder
        self.register_buffer("encoder_pos_emb", self.generate_position_encoding(self.max_seq_len, d_model_encoder), persistent=False)
        self.register_buffer("decoder_pos_emb", self.generate_position_encoding(self.max_seq_len, d_model_decoder), persistent=False)
        self.encoder = TransformerEncoder(
            d_model = d_model_encoder,
            n_head = n_head,
            ffn_hidden = ffn_hidden,
            drop_prob = drop_prob,
            n_layers = n_layers,
            max_seq_len = max_seq_len,
            device = self.device
        )

        self.decoder = TransfomerDecoder(
            d_model_1 = d_model_decoder,
            d_model_2 = d_model_encoder,
            n_head = n_head,
            ffn_hidden = ffn_hidden,
            drop_prob = drop_prob,
            n_layers = n_layers,
            max_seq_len = max_seq_len,
            device = self.device
        )


    """
        encoder_seq:{"input":...,"attention_mask":..}  
        decoder_seq:{"input":...,"attention_mask":..}   attention_mask是batch_mask 用于并行训练
    """

    def forward(self, encoder_seq:dict, decoder_seq:dict, pos_emb = "absolute"):
        # 1 linear transformation
        x_encoder = encoder_seq["input"]
        # x_encoder = self.encoder_linear(encoder_seq["input"])
        if pos_emb == "absolute":
            b,t,f = x_encoder.shape
            x_encoder = x_encoder +self.encoder_pos_emb[:t,].unsqueeze(0).repeat(b,1,1)
        src_padding_mask = encoder_seq["attention_mask"].unsqueeze(2).expand(-1, -1,x_encoder.size(1))
        # src_padding_mask = self.create_padding_mask(encoder_seq["attention_mask"])
        hidden_feather = self.encoder(x_encoder, src_padding_mask, pos_emb)

        x_decoder = decoder_seq["input"]
        # x_decoder = self.decoder_linear(decoder_seq["input"])
        if pos_emb == "absolute":
            b,t,f = x_decoder.shape
            x_decoder = x_decoder + self.decoder_pos_emb[:t,].unsqueeze(0).repeat(b,1,1)
        trg_padding_mask = decoder_seq["attention_mask"]
        causal_mask = self.create_causal_mask(x_decoder)
        output = self.decoder(x_decoder , hidden_feather, padding_mask = trg_padding_mask, causal_mask = causal_mask,pos_emb = pos_emb)
        return output

    def create_causal_mask(self, x):
        b,t,f = x.shape
        mask = torch.ones(b,t,t,dtype=torch.int64).to(self.device)
        for i in range(t):
            mask[:,i,i+1:] = 0
        return mask
    
    def create_padding_mask(self,attention_mask):
        padding_mask = attention_mask.unsqueeze(1).expand(-1, attention_mask.size(1), -1)
        return padding_mask

    def generate_position_encoding(self,max_seq_len,d_model):
        encoding = torch.zeros(max_seq_len, d_model).float().to(self.device)
        encoding.requires_grad = False

        pos = torch.arange(0, max_seq_len, dtype=torch.float32, device=self.device)
        pos = pos.float().unsqueeze(dim=1).to(self.device)

        _2i = torch.arange(0, d_model, step=2).float().to(self.device)

        encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

        return encoding

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension. 

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out

class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, head, 1, 1)
            score = score * mask

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score
    
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

"""
多头注意力机制
d_model_1：Q的特征维度
d_model_2：K、V的特征维度
"""
class MultiHeadAttention(nn.Module):
    def __init__(
        self, 
        d_model_1,
        d_model_2, 
        n_head,
        max_seq_len = 1024,
        device = "cpu",
    ):
        super(MultiHeadAttention, self).__init__()
        self.d_model_1 = d_model_1
        self.d_model_2 = d_model_2
        self.n_head = n_head
        self.max_seq_len = max_seq_len
        self.device = device
        self.set_cos_sin_cache()
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model_1, d_model_1)
        self.w_k = nn.Linear(d_model_2, d_model_1)
        self.w_v = nn.Linear(d_model_2, d_model_1)
        self.w_concat = nn.Linear(d_model_1, d_model_1)

    def forward(self, q, k, v, mask=None,pos_emb = None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        if pos_emb == "rotary":
            b,t,_ = q.shape
            position_ids = torch.arange(t).unsqueeze(0).expand(b, -1)
            cos = self.cos_cached[position_ids]
            sin = self.sin_cached[position_ids]
            q = (q * cos) + (self.rotate_half(q) * sin)
            k = (k * cos) + (self.rotate_half(k) * sin)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out
    
    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

    def set_cos_sin_cache(self,):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.d_model_1, 2).float().to(self.device) / self.d_model_1))
        t = torch.arange(self.max_seq_len, device=self.device, dtype = inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(inv_freq.dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(inv_freq.dtype), persistent=False)

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

class EncoderLayer(nn.Module):
    """
    d_model：特征数量
    ffn_hidden：feed forward隐藏层数量
    n_head：多头数量
    drop_prob：drop比例
    """
    def __init__(self, d_model, ffn_hidden, n_head, max_seq_len,drop_prob,device):
        super(EncoderLayer, self).__init__()
        self.max_seq_len = max_seq_len
        self.device = device
        self.attention = MultiHeadAttention(
            d_model_1 = d_model, 
            d_model_2 = d_model,
            n_head=n_head,
            max_seq_len =  max_seq_len,
            device = self.device
        )
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask, pos_emb = None):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask = src_mask,pos_emb = pos_emb)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)
      
        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x

class DecoderLayer(nn.Module):
    """
    d_model_1：decoder特征数量
    d_model_2:   encoder特征数量
    ffn_hidden：feed forward隐藏层数量
    n_head：多头数量
    drop_prob：drop比例
    """
    def __init__(self, d_model_1,d_model_2, ffn_hidden, n_head, max_seq_len, drop_prob, device):
        super(DecoderLayer, self).__init__()
        self.max_seq_len = max_seq_len
        self.device = device
        self.self_attention = MultiHeadAttention(
            d_model_1 = d_model_1,
            d_model_2 = d_model_1, 
            n_head = n_head
        )
        self.norm1 = LayerNorm(d_model=d_model_1)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(
            d_model_1 = d_model_1, 
            d_model_2 = d_model_2,
            n_head=n_head,
            max_seq_len =  max_seq_len,
            device = self.device
        )
        self.norm2 = LayerNorm(d_model=d_model_1)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model_1, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model_1)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x_decoder, hidden_feather, padding_mask, causal_mask,pos_emb = None):
        # 1. compute self attention
        _x = x_decoder

        x = self.self_attention(q=x_decoder, k=x_decoder, v=x_decoder, mask = causal_mask & padding_mask.unsqueeze(2).expand(-1, -1,x_decoder.size(1)), pos_emb = pos_emb)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if hidden_feather is not None:
            # 3. compute encoder - decoder attention
            _x = x
            x = self.enc_dec_attention(q=x, k=hidden_feather, v=hidden_feather, mask = padding_mask.unsqueeze(2).expand(-1, -1,hidden_feather.size(1)), pos_emb = None)
            
            # 4. add and norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        
        # 6. add and norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x

class TransformerEncoder(nn.Module):
    """
    d_model：特征数量
    ffn_hidden：feed forward隐藏层数量
    n_head：多头数量
    drop_prob：drop比例
    n_layers：EncoderLayer数量
    """
    def __init__(
            self, 
            d_model, 
            ffn_hidden, 
            n_head, 
            n_layers, 
            drop_prob,
            max_seq_len,
            device
        ):
        super().__init__()
        self.device = device
        self.max_seq_len = max_seq_len
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,
                    ffn_hidden=ffn_hidden,
                    n_head=n_head,
                    drop_prob=drop_prob,
                    max_seq_len = max_seq_len,
                    device = device
                ).to(self.device)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, src_mask = None, pos_emb = None):
        for layer in self.layers:
            x = layer(x, src_mask,pos_emb)

        return x

class TransfomerDecoder(nn.Module):
    """
    d_model_1：decoder特征数量
    d_model_2:   encoder特征数量
    ffn_hidden：feed forward隐藏层数量
    n_head：多头数量
    drop_prob：drop比例
    n_layers：EncoderLayer数量
    """
    def __init__(self, d_model_1,d_model_2, ffn_hidden, n_head, n_layers, drop_prob,max_seq_len ,device):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.device = device
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model_1 = d_model_1,
                    d_model_2 = d_model_2,
                    ffn_hidden = ffn_hidden,
                    n_head = n_head,
                    max_seq_len = max_seq_len,
                    drop_prob = drop_prob,
                    device = device
                ).to(self.device)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x_decoder , hidden_feather, padding_mask = None, causal_mask = None,pos_emb = None):

        for layer in self.layers:
            trg = layer(x_decoder, hidden_feather, padding_mask, causal_mask,pos_emb)

        return trg
    