import torch
from torch import nn
import math
import torch.nn.functional as F
from transformer import *

def initialize_weights(m):
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)


if __name__ == "__main__":
    enc_voc_size = 5893
    dec_voc_size = 7853
    src_pad_idx = 1
    trg_pad_idx = 1
    trg_src_idx = 2
    batch_size = 128
    maxlen = 1024
    model_dim = 512
    n_layers = 3
    num_heads = 2
    hidden = 1024
    drop_prob = 0.1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Transformer(
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        enc_voc_size=enc_voc_size,
        dec_voc_size=dec_voc_size,
        maxlen=maxlen,
        model_dim=model_dim,
        hidden=hidden,
        num_head=num_heads,
        n_layers=n_layers,
        drop_prob=drop_prob,
        device=device,
    ).to(device)

    model.apply(initialize_weights)
    src = torch.load("tensor_src.pt", weights_only=True)
    src = torch.cat((src, torch.ones(src.shape[0], 2, dtype=torch.int)), dim=-1).to(device)
    trg = torch.load("tensor_trg.pt", weights_only=True).to(device)

    result = model(src, trg)
    print(result, result.shape)
