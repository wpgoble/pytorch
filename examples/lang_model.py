import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

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
