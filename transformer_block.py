import torch.nn as nn

from attention import MultiHeadAttention

class TransformerBlock(nn.Module):
	"""
	Repeating transformer components in encoder and decoder module
	section 3.1 and 3.2, Figure 1
	"""
	def __init__(self, 
				embed_size=512,
				heads=8,
				dropout=0.1,
				forward_expansion=4,
				):
		super(TransformerBlock, self).__init__()

		self.att = MultiHeadAttention(embed_size, heads=heads)
		self.dropout = nn.Dropout(dropout)
		self.ffn = nn.Sequential(
						nn.Linear(embed_size, embed_size*forward_expansion),
						nn.ReLU(),
						nn.Linear(embed_size*forward_expansion, embed_size)
					)

	def forward(self, query, key, value, mask=None):
		x = self.att(query, key, value, mask=mask)
		x = self.dropout(query + x)

		intermediate = self.ffn(x)
		x = self.dropout(intermediate + x)

		return x
