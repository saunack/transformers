import torch
from transformer import Transformer

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    N = 15

    src_vocab_size = 10
    src_pad_idx = 0

    inp_size = 38
    x = torch.randint(1,src_vocab_size, (N,inp_size)).to(device)
    # add pads randomly at the end
    input_pads_idx = inp_size - 1 - torch.randint(0,5,(N,))
    input_pads_idx = input_pads_idx.unsqueeze(1).expand(N,inp_size).to(device)
    indices = torch.arange(inp_size).expand(N,inp_size).to(device)
    x[indices<=input_pads_idx] = src_pad_idx

    tgt_vocab_size = 10
    tgt_pad_idx = 0
    out_size = 45

    output_pads_idx = torch.randint(0,5,(N,))
    trg = torch.randint(1,tgt_vocab_size, (N,out_size)).to(device)
    # add pads randomly at the end
    output_pads_idx = out_size - 1 - torch.randint(0,5,(N,))
    output_pads_idx = output_pads_idx.unsqueeze(1).expand(N,out_size).to(device)
    indices = torch.arange(out_size).expand(N,out_size).to(device)
    trg[indices<=output_pads_idx] = tgt_pad_idx

    model = Transformer(src_vocab_size, tgt_vocab_size, src_pad_idx, tgt_pad_idx, device=device).to(
        device
    )
    out = model(x, trg)
    print(out.shape)

