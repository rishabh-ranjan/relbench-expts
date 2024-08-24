from einops import rearrange, einsum
import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, e):
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = rearrange(xq, "b i (h d) -> b h i d", h=self.num_heads)
        xk = rearrange(xk, "b i (h d) -> b h i d", h=self.num_heads)
        xv = rearrange(xv, "b i (h d) -> b h i d", h=self.num_heads)

        sqrt_d = xq.size(-1) ** 0.5
        a = einsum(xq, xk, "b h i d, b h j d -> b h i j")
        a = F.softmax(a / sqrt_d, dim=-1)
        x = einsum(a, xv, "b h i j, b h j d -> b h i d")
        # TODO: add e

        x = rearrange(x, "b h i d -> b i (h d)")
        x = self.wo(x)

        return x


class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.norm_attn = nn.RMSNorm(d_model)
        self.norm_ffn = nn.RMSNorm(d_model)
        self.attn = Attention(d_model, num_heads)
        self.ffn = FFN(d_model, d_ff)

    def forward(self, x, e):
        x = x + self.attn(self.norm_attn(x), e)
        x = x + self.ffn(self.norm_ffn(x))
        return x


class TransformerCore(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff):
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(TransformerBlock(d_model, num_heads, d_ff))

        self.norm_out = nn.RMSNorm(d_model)

    def forward(self, x, e):
        for layer in self.layers:
            x = layer(x, e)
        x = self.norm_out(x)
        return x


class RelTransformer(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        d_ff,
        node_types,
    ):
        super().__init__()

        self.node_type_emb = nn.Embedding(len(node_types), d_model)
        self.transformer = TransformerCore(num_layers, d_model, num_heads, d_ff)

    def forward(self, x_dict, edge_index_dict):
        for node_type in x_dict.keys():
            x_dict[node_type] += self.node_type_emb(node_type)

        x = rearrange(x_dict, "n i d -> n i d")

        x = self.transformer(x, e=None)

        return x
