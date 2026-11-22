import math

import torch
from torch import nn
import torch.nn.functional as F


def attention(q,k,v): # scaled dot product attention softmax(q k^T / sqrt(d)) @ v
    d = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)
    weights = scores.softmax(dim=-1)
    return weights @ v , weights

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0, "dim must split evenly across heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        out, weights = attention(q, k, v)
        out = out.transpose(1,2).reshape(B, N, D)
        return self.proj(out), weights


if __name__ == "__main__":
    torch.manual_seed(42)

    # single head vs pytorch built in function
    q, k, v = torch.randn(3, 2, 5, 8)
    mine, _ = attention(q, k, v)
    theirs = F.scaled_dot_product_attention(q, k, v)
    print("single-head max difference:", (mine - theirs).abs().max().item())
    assert torch.allclose(mine, theirs, atol=1e-6)

    # multi head vs  nn.MultiheadAttention using the same weights
    dim, heads = 64, 4
    x = torch.randn(2, 65, dim)          # 2 images and 65 tokens
    mine_mha = MultiHeadSelfAttention(dim, heads)
    torch_mha = nn.MultiheadAttention(dim, heads, batch_first=True)
    with torch.no_grad():
        torch_mha.in_proj_weight.copy_(mine_mha.qkv.weight)
        torch_mha.in_proj_bias.copy_(mine_mha.qkv.bias)
        torch_mha.out_proj.weight.copy_(mine_mha.proj.weight)
        torch_mha.out_proj.bias.copy_(mine_mha.proj.bias)

    mine_out, mine_w = mine_mha(x)
    torch_out, torch_w = torch_mha(x, x, x)
    print("multi-head max difference:", (mine_out - torch_out).abs().max().item())
    print("weights max difference:", (mine_w.mean(dim=1) - torch_w).abs().max().item())
    assert torch.allclose(mine_out, torch_out, atol=1e-6)
    assert torch.allclose(mine_w.mean(dim=1), torch_w, atol=1e-6)
    assert torch.allclose(mine_w.sum(dim=-1), torch.ones(2, heads, 65))
    print("All attention checks passed")