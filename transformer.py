import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder

from utils import create_autogression_mask

class Transformer(nn.Module):
	def __init__(self,
                src_vocab_size,
                tgt_vocab_size,
                src_pad_token=0,
                tgt_pad_token=0,
                embed_size=512,
                heads=8,
                dropout=0.1,
                forward_expansion=4,
                N=6,
                device='cuda',
				):
		super(Transformer, self).__init__()

		self.embed_size = embed_size
		self.src_pad_token = src_pad_token
		self.tgt_pad_token = tgt_pad_token
		self.device = device
		self.encoder = Encoder(
				src_vocab_size,
                embed_size=embed_size,
                heads=heads,
                dropout=dropout,
                forward_expansion=forward_expansion,
                N=N,
                device=device,
                )

		self.decoder = Decoder(
				tgt_vocab_size,
                embed_size=embed_size,
                heads=heads,
                dropout=dropout,
                forward_expansion=forward_expansion,
                N=N,
                device=device,
                )

	def create_pad_mask(self, src, pad_tok):
		# create mask by comparing tokens with padding token
		# shape of src: N * seq_len
		# shape required for attention mask: N * 1 * 1 * seq_len
		"""
		N * 1 * seq_len * 1 will replicate incorrectly during masking and will mask all attention scores in a row
		or will keep all scores. For example, create any (N,) tensor and unsqueeze along different axes followed by masking a (N,N) tensor
		"""
		mask = (src != pad_tok).unsqueeze(1).unsqueeze(2)
		return mask.to(self.device)


	def forward(self, input_tokens, output_tokens):
		# input_tokens shape: N * input_seq_len
		input_pad_mask = self.create_pad_mask(input_tokens, self.src_pad_token)
		encoder_out = self.encoder(input_tokens, mask=input_pad_mask)

		# output_tokens shape: N * output_seq_len
		output_seq_len = output_tokens.shape[-1]
		N = output_tokens.shape[0]
		output_pad_mask = self.create_pad_mask(output_tokens, self.tgt_pad_token)
		autogression_mask = create_autogression_mask(output_seq_len, device=self.device)
		# combine pad and lookahead mask. Can use autogression_mask directly too since pads appear after [SEP] and are always contiguous
		# decoder_mask = autogression_mask
		decoder_mask = torch.matmul(output_pad_mask.expand(N,1,output_seq_len,output_seq_len).float(),
									autogression_mask.float().expand(N,1,output_seq_len, output_seq_len))
		# combine output pad and input pad. Only input pad will work since we don't care about attention for padded output tokens
		# shapes: output pad: N,1,1,q input pad: N,1,1,k. Required: N,1,q,k
		# encoder_mask = input_pad_mask
		encoder_mask = torch.matmul(output_pad_mask.reshape(N,1,output_seq_len,1).float(),
									input_pad_mask.float())
		x = self.decoder(output_tokens, encoder_out, decoder_mask=decoder_mask, encoder_mask=encoder_mask)

		return x

