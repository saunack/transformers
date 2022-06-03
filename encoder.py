import torch.nn as nn
import torch

from transformer_block import TransformerBlock
from utils import get_positional_embedding

def Encoder(nn.Module):
    def __init__(self,
                vocab_size, 
                embed_size=512,
                heads=8,
                dropout=0.1,
                forward_expansion=4,
                N=6,
                device='cuda',
                ):
        super(Encoder).__init__()

        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.embed_size = embed_size
        self.transformer_blocks = [TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(N)]
        self.device = device


    def forward(self, x, mask):
        # add positional encodings
        embedding = self.word_embedding(x)
        N = embedding.shape[-2]
        positional = get_positional_embedding(N, self.embed_size, device=self.device)

        x = positional + embedding
        # run through each transformer block
        for block in self.transformer_blocks:
            x = block(x,x,x, mask)

        return x
