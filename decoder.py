import torch.nn as nn

from attention import MultiHeadAttention
from transformer_block import TransformerBlock

def Decoder(nn.Module):
    def __init__(super, 
                embed_size=512,
                heads=8,
                dropout=0.1,
                N=6,
                ):
        super(Decoder).__init__()

        self.transformer_blocks = [TransformerBlock(embed_size, heads, dropout) for _ in range(N)]
        self.masked_attention = [MultiHeadAttention(embed_size, heads, dropout) for _ in range(N)]

    def forward(self, query, key, value, mask=None):

        return x
