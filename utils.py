import torch


def get_positional_embedding(self, N, embed_size, device='cuda'):
    """
    Positional sine and cosine embeddings
    Section 3.5 of the paper
    PE(pos,2i) = sin(pos/10000^2i/d_model )
    PE(pos,2i+1) = cos(pos/10000^ 2i/d_model )
    for i in [0,d_model)
        pos in [0,input_length)
    where 
    d_model: embedding dimesions
    """
    # shape : embed_size, 1
    positional_exp_base = torch.full((1,embed_size),10000)
    # get sequence of [[0,0,2,2,4,4,...,embed_size-embed_size%2,embed_size-embed_size%2]]/embed_size
    # shape : 1, embed_size
    positional_exp_expo = torch.arange(start=0,end=embed_size+1,step=2).\
                                repeat_interleave(2)[:embed_size].unsqueeze(dim=0)/embed_size
    # shape : 1, embed_size
    positional_den = torch.reciprocal(torch.pow(positional_exp_base, positional_exp_expo))
    # shape : N, 1
    positional_num = torch.arange(start=0,end=N,step=1).unsqueeze(dim=1).type(torch.FloatTensor)
    # shape: N, embed_size
    positional = torch.matmul(positional_num, positional_den)
    # sine and cosine at even and odd dimensions
    even_index = torch.arange(0,embed_size,2)
    odd_index = torch.arange(1,embed_size+1,2)
    positional[:,even_index] = torch.sin(positional[:,even_index])
    positional[:,odd_index] = torch.cos(positional[:,odd_index])
    return positional


def create_mask(self, N, embed_size):
    return torch.tril((N,embed_size))