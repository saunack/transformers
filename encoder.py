import torch.nn as nn
import torch

from transformer_block import TransformerBlock
from utils import get_positional_embedding

def Encoder(nn.Module):
    def __init__(super, 
                embed_size=512,
                heads=8,
                dropout=0.1,
                N=6,
                ):
        super(Encoder).__init__()

        self.embed_size = embed_size
        self.transformer_blocks = [TransformerBlock(embed_size, heads, dropout) for _ in range(N)]


    def forward(self, embedding):
        # add positional encodings
        N = embedding.shape[-2]
        assert(self.embed_size = embedding.shape[-1])
        positional = get_positional_embedding(N, self.embed_size)

        x = positional + embedding
        # run through each transformer block
        for block in self.transformer_blocks:
            x = block(x,x,x)

        return x
