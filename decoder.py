import torch.nn as nn

from attention import MultiHeadAttention
from transformer_block import TransformerBlock
from utils import get_positional_embedding

class Decoder(nn.Module):
    """
    Decoder section of the transformer paper
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
        super(Decoder, self).__init__()

        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.embed_size = embed_size
        self.masked_attention = [MultiHeadAttention(embed_size, heads) for _ in range(N)]
        self.transformer_blocks = [TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(N)]
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        self.device = device

    def forward(self, x, encoder_out, decoder_mask, encoder_mask):
        # convert to embedding space
        embedding = self.word_embedding(x)
        N = embedding.shape[-2]

        # add positional encodings
        positional = get_positional_embedding(N, self.embed_size, device=self.device)
        x = positional + embedding
        # run through each transformer block
        for i in range(len(self.transformer_blocks)):
            # mask for autogression 
            mha = self.masked_attention[i](x,x,x,mask=decoder_mask)
            x = self.dropout(x + mha)
            # pads on both source and target mask
            x = self.transformer_blocks[i](x,encoder_out,encoder_out, mask=encoder_mask)
 
        x = self.softmax(self.fc_out(x))

        return x
