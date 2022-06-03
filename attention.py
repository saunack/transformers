import torch
import torch.nn as nn

def MultiHeadAttention(nn.Module):
	"""
	multi head attention component
	section 3.2.1, 3.2.2, Figure 2 of the paper
	"""
	def __init__(self, embed_size, heads=8):
		super(MultiHeadAttention).__init__()
		self.embed_size = embed_size
		self.heads = heads
		self.head_dims = embed_size//heads

		assert(self.head*self.head_dims == self.embed_size)

		# creating the layers

		# each head is produced via projections
		self.queries = nn.Linear(self.embed_size, self.embed_size)
		self.keys = nn.Linear(self.embed_size, self.embed_size)
		self.values = nn.Linear(self.embed_size, self.embed_size)
		
		# final output connection
		self.linear = nn.Linear(self.embed_size, self.embed_size)

	def forward(self, query, key, value, mask=None):
		"""
		query: equal to key for encoder, derived from output for decoder
		key: always equal to value dimensions
		value: always equal to key dimensions
		"""
		N = query.shape[0]
		query_len, key_len, value_len = query.shape[-1], key.shape[-1], value.shape[-1]

		# pass through projection
		q = self.query(query)
		k = self.key(key)
		v = self.value(value)

		# logical separation via creation of separate heads
		# reference: https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853
		"""
		This is equivalent to having self.heads * linear layers mapping from embed_size to head_dims.
		This is equivalent because during attention, in both cases, each embedding dimesion is used to create the heads 
		 and only corresponding heads from q,k,v are responsible for attention.
		Reshaping first and then performing a linear projection is incorrect because then the projections use limited sections
		 of embeddings for the attention 
		By doing a logical separation, we only need to have one matrix common for all heads for each q,v,k
		"""

		q = q.reshape(N, self.heads, query_len, self.head_dims)
		k = q.reshape(N, self.heads, key_len, self.head_dims)
		v = q.reshape(N, self.heads, value_len, self.head_dims)

		
		# scaled dot product attention
		# q: (N, heads, query_length, head_dims)
		# k: (N, heads, key_length, head_dims)
		# attention: (N, heads, query_length, key_length)
		attention = torch.matmul(q, k.reshape(N, self.heads, self.head_dims, key_len))
		# either use matmul or einsum. resulting attention shape is the same
		# attention = torch.einsum('nhqd,nhkd->nhqk',q, k)
		attention = attention/(self.head_dims**0.5)
		

		if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(attention, dim=-1)

		# attention: (N, heads, query_length, key_length)
		# v: (N, heads, value_length, head_dims)
		# output: (N, heads, query_length, head_dims)
        output = torch.einsum('nhqk,nhkd->nhqd',attention,v)
        output = output.reshape(N, self.heads, quer_len, embed_size)

        return self.linear(output)

