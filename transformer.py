import torch.nn as nn

from encoder import Encoder
from decoder import Decoder

from utils import create_mask

def Transformer(nn.Module):
	def __init__(self,
                src_vocab_size,
                tgt_vocab_size,
                embed_size=512,
                heads=8,
                dropout=0.1,
                forward_expansion=4,
                N=6,
                device='cuda',
				):
		super(Transformer).__init__()

		self.embed_size = embed_size
		self.encoder = Encoder(
				src_vocab_size,
                embed_size=512,
                heads=8,
                dropout=0.1,
                forward_expansion=4,
                N=6,
                device='cuda',
                )

		self.decoder = Decoder(
				tgt_vocab_size,
                embed_size=512,
                heads=8,
                dropout=0.1,
                forward_expansion=4,
                N=6,
                device='cuda',
                )

	def forward(self, input_words, output_words):
		x = self.encoder(input_words, mask=None)
		mask = create_mask(output_words.shape[-1], self.embed_size)
		x = self.decoder(output_words, x, mask=None)

		return nn.Softmax(x)