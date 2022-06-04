import torch.nn as nn
import torch

from transformer_block import TransformerBlock
from utils import get_positional_embedding

class Encoder(nn.Module):
    """
    Encoder section of the transformer paper
    Section 3.1, 3.3, 3.4, Figure 2 of the paper
    """
    def __init__(self,
                vocab_size, 
                embed_size=512,
                heads=8,
                dropout=0.1,
                forward_expansion=4,
                N=6,
                device='cuda',
                ):
        super(Encoder, self).__init__()

        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.embed_size = embed_size
        self.transformer_blocks = [TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(N)]
        self.device = device


    def forward(self, x, mask):
        # convert to embedding space
        embedding = self.word_embedding(x)
        N = embedding.shape[-2]

        # add positional encodings
        positional = get_positional_embedding(N, self.embed_size, device=self.device)
        x = positional + embedding
        # run through each transformer block
        for block in self.transformer_blocks:
            # mask for padding due to size N * max_token * embedding size
            x = block(x,x,x,mask)

        return x
