import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

class TransformerModel(nn.Module):
    def __init__(self, ntoken:int, d_model:int, nhead:int, d_hid:int, 
                    nlayers:int, dropout:float = 0.5):
        super().__init__()
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src:Tensor, src_mask:Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_length, batch_size]
            src_mask: Tensor, shape [seq_length, seq_length]
        
        Returns:
            output Tensor of shape [seq_length, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

def generate_square_subsequent_mask(sz:int) -> Tensor:
    """ Generates an upper-triangular matrix of -inf, with zeros on diag.  """
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal = 1)

"""
PositionalEncoding module injects some information about the relative or 
absolute position of the tokens in the sequence. The positional encodings have 
the same dimension as the embeddings so that the two can be summed. Here, we 
use sine and cosine functions of different frequencies.
"""
class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, dropout:float = 0.1, max_len:int = 5_000):
        super().__init__()
        self.dropout = nn.Dropout(p = dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
            (-math.log(-10_000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forwarrd(self, x:Tensor) -> Tensor:
        """
        Args: 
            x: Tensor, shape [seq_length, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# Load and batch data
train_iter = WikiText2(split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
    """ Converts raw text into a float Tensor """
    data = [torch.tensor(vocab(tokenizer(item)), d_type = torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

train_iter, val_iter, test_iter = WikiText2()
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def batchify(data:Tensor, bsz:int) -> Tensor:
    """
    Divides the data into bsz separate sequences, removing extra elements that 
    wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size
    
    Returns:
        Tensor of shape [N // bsz, bsz]
    """

    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)