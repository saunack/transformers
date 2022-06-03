import torch.nn as nn

from attention import MultiHeadAttention

def TransformerBlock(nn.Module):
	def __init__(super, 
				embed_size=512,
				heads=8,
				dropout=0.1,
				):
		super(TransformerBlock).__init__()

		self.att = MultiHeadAttention(embed_size, heads=heads)
		self.dropout = nn.Dropout(dropout)
		self.ffn = nn.Linear(embed_size, embed_size)

	def forward(self, query, key, value, mask=None):
		x = self.att(query, key, value, mask=mask)
		x = self.dropout(value + x)

		intermediate = self.ffn(x)
		x = self.dropout(intermediate + x)

		return x
