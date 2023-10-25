from typing import List, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from einops import rearrange, repeat

# from vit_pytorch.vit_face import ViT_face
# from vit_pytorch.vits_face import ViTs_face

MIN_NUM_PATCHES = 16


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max
        #embed()
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)

        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, pool_length):
        super().__init__()
        self.pool_length = pool_length
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        xs = []
        for layer_i, (attn, ff) in enumerate(self.layers):
            x = attn(x, mask = mask)
            x = ff(x)
            if layer_i >= len(self.layers)-self.pool_length:
                xs.append(x)
        return xs
                    
class AttentiveStatisticsPooling(nn.Module):
    """Attentive statistics pooling.
    """
    def __init__(self, channels: int, bottleneck: int):
        """Initializer.
        Args:
            channels: size of the input channels.
            bottleneck: size of the bottleneck.
        """
        super().__init__()
        # nonlinear=Tanh
        # ref: https://github.com/KrishnaDN/Attentive-Statistics-Pooling-for-Deep-Speaker-Embedding
        # ref: https://github.com/TaoRuijie/ECAPA-TDNN
        self.attention = nn.Sequential(
            nn.Conv1d(channels, bottleneck, 1),
            nn.Tanh(),
            nn.Conv1d(bottleneck, channels, 1),
            nn.Softmax(dim=-1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Pooling with weighted statistics.
        Args:
            inputs: [torch.float32; [B, C, T]], input tensors,
                where C = `channels`.
        Returns:
            [torch.float32; [B, C x 2]], weighted statistics.
        """
        # [B, C, T]
        weights = self.attention(inputs)
        # [B, C]
        mean = torch.sum(weights * inputs, dim=-1)
        var = torch.sum(weights * inputs ** 2, dim=-1) - mean ** 2
        # [B, C x 2], for numerical stability of square root
        return torch.cat([mean, (var + 1e-7).sqrt()], dim=-1)
    
    
class MultiheadAttention(nn.Module):
    """Multi-head scaled dot-product attention.
    """
    def __init__(self,
                 keys: int,
                 values: int,
                 queries: int,
                 out_channels: int,
                 hiddens: int,
                 heads: int):
        """Initializer.
        Args:
            keys, valeus, queries: size of the input channels.
            out_channels: size of the output channels.
            hiddens: size of the hidden channels.
            heads: the number of the attention heads.
        """
        super().__init__()
        assert hiddens % heads == 0, \
            f'size of hiddens channels(={hiddens}) should be factorized by heads(={heads})'
        self.channels, self.heads = hiddens // heads, heads
        self.proj_key = nn.Conv1d(keys, hiddens, 1)
        self.proj_value = nn.Conv1d(values, hiddens, 1)
        self.proj_query = nn.Conv1d(queries, hiddens, 1)
        self.proj_out = nn.Conv1d(hiddens, out_channels, 1)

    def forward(self,
                keys: torch.Tensor,
                values: torch.Tensor,
                queries: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Transform the inputs.
        Args:
            keys: [torch.float32; [B, keys, S]], attention key.
            values: [torch.float32; [B, values, S]], attention value.
            queries: [torch.float32; [B, queries, T]], attention query.
            mask: [torch.float32; [B, S, T]], attention mask, 0 for paddings.
        Returns:
            [torch.float32; [B, out_channels, T]], transformed outputs.
        """
        # B, T
        bsize, _, querylen = queries.shape
        # S
        keylen = keys.shape[-1]
        assert keylen == values.shape[-1], 'lengths of key and value are not matched'
        # [B, H, hiddens // H, S]
        keys = self.proj_key(keys).view(bsize, self.heads, -1, keylen)
        values = self.proj_value(values).view(bsize, self.heads, -1, keylen)
        # [B, H, hiddens // H, T]
        queries = self.proj_query(queries).view(bsize, self.heads, -1, querylen)
        # [B, H, S, T]
        score = torch.matmul(keys.transpose(2, 3), queries) * (self.channels ** -0.5)
        if mask is not None:
            score.masked_fill_(~mask[:, None, :, :1].to(torch.bool), -np.inf)
        # [B, H, S, T]
        weights = torch.softmax(score, dim=2)
        # [B, out_channels, T]
        out = self.proj_out(
            torch.matmul(values, weights).view(bsize, -1, querylen))
        if mask is not None:
            out = out * mask[:, :1]
        return out


class ViTs_face(nn.Module):
    def __init__(self, *, out_channels, image_size, patch_size, ac_patch_size,
                         pad, dim, depth, heads, mlp_dim, img_channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.,
                         pool_length = 3, channels, hiddens, bottleneck):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = img_channels * ac_patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        # assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.patch_size = patch_size
        self.soft_split = nn.Unfold(kernel_size=(ac_patch_size, ac_patch_size), stride=(self.patch_size, self.patch_size), padding=(pad, pad))


        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, pool_length)

        self.conv1x1 = nn.Sequential(
            nn.Conv1d(pool_length * channels, hiddens, 1),
            nn.ReLU())
        self.pool = nn.Sequential(
            AttentiveStatisticsPooling(hiddens, bottleneck),
            nn.BatchNorm1d(hiddens * 2),
            nn.Linear(hiddens * 2, out_channels),
            nn.BatchNorm1d(out_channels))

    def forward(self, img, label= None , mask = None):
        p = self.patch_size
        x = self.soft_split(img).transpose(1, 2)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        xs = self.transformer(x, mask)
        
        # [B, H, T]
        mfa = self.conv1x1(torch.cat(xs, dim=2).transpose(1,2))
        
        # [B, O]
        global_ = F.normalize(self.pool(mfa), p=2, dim=-1)
        return global_, mfa
            
        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        # x = self.to_latent(x)
        # emb = self.mlp_head(x)
        # if label is not None:
            # x = self.loss(emb, label)
            # return x, emb
        # else:
        # return emb
        

class FimbreEncoder(nn.Module):
    def __init__(self,
                 img_size: int,
                 in_channels: int,                 
                 out_channels: int,
                 channels: int,
                 patch_size: int,
                 ac_patch_size: int,
                 pad: int,
                 dim: int,
                 depth: int,
                 heads: int,
                 mlp_dim: int,
                 dim_head: int,
                 dropout: float,
                 emb_dropout: float,
                 pool_length: int,
                 bottleneck: int,
                 hiddens: int,
                 latent: int,
                 fimber: int,
                 tokens: int,
                 contents: int,
                 slerp: float
                 ):
        """Initializer.
        Args:
            in_channels: size of the input channels.
            out_channels: size of the output embeddings.
            channels: size of the major states.
            prekernels: size of the convolutional kernels before feed to SERes2Block.
            scale: the number of the resolutions, for SERes2Block.
            kernels: size of the convolutional kernels, for SERes2Block.
            dilations: dilation factors.
            bottleneck: size of the bottleneck layers,
                both SERes2Block and AttentiveStatisticsPooling.
            hiddens: size of the hidden channels for attentive statistics pooling.
            latent: size of the timber latent query.
            timber: size of the timber tokens.
            tokens: the number of the timber tokens.
            heads: the number of the attention heads, for timber token block.
            contents: size of the content queries.
            slerp: weight value for spherical interpolation.
        """
        super().__init__()
        # vit_face blocks
        self.vit_face = ViTs_face(out_channels=out_channels, image_size=img_size, patch_size=patch_size, ac_patch_size=ac_patch_size,
                                pad=pad, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, img_channels=in_channels, dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout,
                                pool_length=pool_length, channels=channels, hiddens=hiddens, bottleneck=bottleneck)
        # self.vit_face = Vits_face(num_class=192, image_size=112, patch_size=8, ac_patch_size=12,
                                # pad=4, dim=512, depth=20, heads=8, mlp_dim=2048, img_channels=3, dim_head=64, dropout=0.1, emb_dropout=0.1,
                                # pool_length=3, channels, hiddens, bottleneck, out_channels)       
                                                         
        # NANSY++, timber=128
        # unknown `heads`
        self.pre_mha = MultiheadAttention(
            hiddens, hiddens, latent, latent, latent, heads)
        self.post_mha = MultiheadAttention(
            hiddens, hiddens, latent, fimber, latent, heads)
        # time-varying timber encoder
        self.fimber_key = nn.Parameter(torch.randn(1, fimber, tokens))
        self.sampler = MultiheadAttention(
            fimber, fimber, contents, fimber, latent, heads)
        self.proj = nn.Conv1d(fimber, out_channels, 1)
        # unknown `slerp`
        assert 0 <= slerp <= 1, f'value slerp(={slerp:.2f}) should be in range [0, 1]'
        self.slerp = slerp

    def forward(self, inputs: torch.Tensor, fimber_query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the x-vectors from the input sequence.
        Args:
            inputs: [torch.float32; [B, in_channels, T]], input sequences,
        Returns:
            [torch.float32; [B, out_channels]], global x-vectors,
            [torch.float32; [B, timber, tokens]], timber token bank.
        """
        global_, mfa = self.vit_face(inputs)
        # B
        bsize, _ = global_.shape
        # [B, latent, tokens]
        query = fimber_query.repeat(bsize, 1, 1)
        # [B, latent, tokens]
        query = self.pre_mha.forward(mfa, mfa, query) + query
        # [B, timber, tokens]
        local = self.post_mha.forward(mfa, mfa, query)
        # [B, out_channels], [B, timber, tokens]
        return global_, local

    def sample_fimber(self,
                      contents: torch.Tensor,
                      global_: torch.Tensor,
                      tokens: torch.Tensor,
                      eps: float = 1e-5) -> torch.Tensor:
        """Sample the timber tokens w.r.t. the contents.
        Args:
            contents: [torch.float32; [B, contents, T]], content queries.
            global_: [torch.float32; [B, out_channels]], global x-vectors, L2-normalized.
            tokens: [torch.float32; [B, timber, tokens]], timber token bank.
            eps: small value for preventing train instability of arccos in slerp.
        Returns:
            [torch.float32; [B, out_channels, T]], time-varying timber embeddings.
        """
        # [B, timber, tokens]
        key = self.fimber_key.repeat(contents.shape[0], 1, 1)
        # [B, timber, T]
        sampled = self.sampler.forward(key, tokens, contents)
        # [B, out_channels, T]
        sampled = F.normalize(self.proj(sampled), p=2, dim=1)
        # [B, 1, T]
        theta = torch.matmul(global_[:, None], sampled).clamp(-1 +  eps, 1 - eps).acos()
        # [B, 1, T], slerp
        # clamp the theta is not necessary since cos(theta) is already clampped
        return (
            torch.sin(self.slerp * theta) * sampled
            + torch.sin((1 - self.slerp) * theta) * global_[..., None]) / theta.sin()