import torch
import torch.nn as nn
import math
import warnings
import numpy as np
from collections import OrderedDict

from model.motionbert.drop import DropPath


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., st_mode='vanilla'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.mode = st_mode
        if self.mode == 'parallel':
            self.ts_attn = nn.Linear(dim * 2, dim * 2)
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_count_s = None
        self.attn_count_t = None

    def forward(self, x, seqlen=1):
        B, N, C = x.shape

        if self.mode == 'series':
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            x = self.forward_spatial(q, k, v)
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            x = self.forward_temporal(q, k, v, seqlen=seqlen)
        elif self.mode == 'parallel':
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            x_t = self.forward_temporal(q, k, v, seqlen=seqlen)
            x_s = self.forward_spatial(q, k, v)

            alpha = torch.cat([x_s, x_t], dim=-1)
            alpha = alpha.mean(dim=1, keepdim=True)
            alpha = self.ts_attn(alpha).reshape(B, 1, C, 2)
            alpha = alpha.softmax(dim=-1)
            x = x_t * alpha[:, :, :, 1] + x_s * alpha[:, :, :, 0]
        elif self.mode == 'coupling':
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            x = self.forward_coupling(q, k, v, seqlen=seqlen)
        elif self.mode == 'vanilla':
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            x = self.forward_spatial(q, k, v)
        elif self.mode == 'temporal':
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            x = self.forward_temporal(q, k, v, seqlen=seqlen)
        elif self.mode == 'spatial':
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            x = self.forward_spatial(q, k, v)
        else:
            raise NotImplementedError(self.mode)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def reshape_T(self, x, seqlen=1, inverse=False):
        if not inverse:
            N, C = x.shape[-2:]
            x = x.reshape(-1, seqlen, self.num_heads, N, C).transpose(1, 2)
            x = x.reshape(-1, self.num_heads, seqlen * N, C)  # (B, H, TN, c)
        else:
            TN, C = x.shape[-2:]
            x = x.reshape(-1, self.num_heads, seqlen, TN // seqlen, C).transpose(1, 2)
            x = x.reshape(-1, self.num_heads, TN // seqlen, C)  # (BT, H, N, C)
        return x

    def forward_coupling(self, q, k, v, seqlen=8):
        BT, _, N, C = q.shape
        q = self.reshape_T(q, seqlen)
        k = self.reshape_T(k, seqlen)
        v = self.reshape_T(v, seqlen)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = self.reshape_T(x, seqlen, inverse=True)
        x = x.transpose(1, 2).reshape(BT, N, C * self.num_heads)
        return x

    def forward_spatial(self, q, k, v):
        B, _, N, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C * self.num_heads)
        return x

    def forward_temporal(self, q, k, v, seqlen=8):
        B, _, N, C = q.shape
        qt = q.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4)  # (B, H, N, T, C)
        kt = k.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4)  # (B, H, N, T, C)
        vt = v.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4)  # (B, H, N, T, C)

        attn = (qt @ kt.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ vt  # (B, H, N, T, C)
        x = x.permute(0, 3, 2, 1, 4).reshape(B, N, C * self.num_heads)
        return x

    def count_attn(self, attn):
        attn = attn.detach().cpu().numpy()
        attn = attn.mean(axis=1)
        attn_t = attn[:, :, 1].mean(axis=1)
        attn_s = attn[:, :, 0].mean(axis=1)
        if self.attn_count_s is None:
            self.attn_count_s = attn_s
            self.attn_count_t = attn_t
        else:
            self.attn_count_s = np.concatenate([self.attn_count_s, attn_s], axis=0)
            self.attn_count_t = np.concatenate([self.attn_count_t, attn_t], axis=0)


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., mlp_out_ratio=1., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, st_mode='stage_st', att_fuse=False):
        super().__init__()
        # assert 'stage' in st_mode
        self.st_mode = st_mode
        self.norm1_s = norm_layer(dim)
        self.norm1_t = norm_layer(dim)
        self.attn_s = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            st_mode="spatial")
        self.attn_t = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            st_mode="temporal")

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2_s = norm_layer(dim)
        self.norm2_t = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_out_dim = int(dim * mlp_out_ratio)
        self.mlp_s = MLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=mlp_out_dim, act_layer=act_layer,
                         drop=drop)
        self.mlp_t = MLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=mlp_out_dim, act_layer=act_layer,
                         drop=drop)
        self.att_fuse = att_fuse
        if self.att_fuse:
            self.ts_attn = nn.Linear(dim * 2, dim * 2)

    def forward(self, x, seqlen=1):
        if self.st_mode == 'stage_st':
            x = x + self.drop_path(self.attn_s(self.norm1_s(x), seqlen))
            x = x + self.drop_path(self.mlp_s(self.norm2_s(x)))
            x = x + self.drop_path(self.attn_t(self.norm1_t(x), seqlen))
            x = x + self.drop_path(self.mlp_t(self.norm2_t(x)))
        elif self.st_mode == 'stage_ts':
            x = x + self.drop_path(self.attn_t(self.norm1_t(x), seqlen))
            x = x + self.drop_path(self.mlp_t(self.norm2_t(x)))
            x = x + self.drop_path(self.attn_s(self.norm1_s(x), seqlen))
            x = x + self.drop_path(self.mlp_s(self.norm2_s(x)))
        elif self.st_mode == 'stage_para':
            x_t = x + self.drop_path(self.attn_t(self.norm1_t(x), seqlen))
            x_t = x_t + self.drop_path(self.mlp_t(self.norm2_t(x_t)))
            x_s = x + self.drop_path(self.attn_s(self.norm1_s(x), seqlen))
            x_s = x_s + self.drop_path(self.mlp_s(self.norm2_s(x_s)))
            if self.att_fuse:
                #             x_s, x_t: [BF, J, dim]
                alpha = torch.cat([x_s, x_t], dim=-1)
                BF, J = alpha.shape[:2]
                # alpha = alpha.mean(dim=1, keepdim=True)
                alpha = self.ts_attn(alpha).reshape(BF, J, -1, 2)
                alpha = alpha.softmax(dim=-1)
                x = x_t * alpha[:, :, :, 1] + x_s * alpha[:, :, :, 0]
            else:
                x = (x_s + x_t) * 0.5
        else:
            raise NotImplementedError(self.st_mode)
        return x


class DSTformer(nn.Module):
    def __init__(self, dim_in=3, dim_out=3, dim_feat=256, dim_rep=512,
                 depth=5, num_heads=8, mlp_ratio=4,
                 num_joints=17, maxlen=243,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, att_fuse=True):
        super().__init__()
        self.dim_out = dim_out
        self.dim_feat = dim_feat
        self.joints_embed = nn.Linear(dim_in, dim_feat)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks_st = nn.ModuleList([
            Block(
                dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                st_mode="stage_st")
            for i in range(depth)])
        self.blocks_ts = nn.ModuleList([
            Block(
                dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                st_mode="stage_ts")
            for i in range(depth)])
        self.norm = norm_layer(dim_feat)
        if dim_rep:
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(dim_feat, dim_rep)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()
        self.head = nn.Linear(dim_rep, dim_out) if dim_out > 0 else nn.Identity()
        self.temp_embed = nn.Parameter(torch.zeros(1, maxlen, 1, dim_feat))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        trunc_normal_(self.temp_embed, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)
        self.att_fuse = att_fuse
        if self.att_fuse:
            self.ts_attn = nn.ModuleList([nn.Linear(dim_feat * 2, 2) for i in range(depth)])
            for i in range(depth):
                self.ts_attn[i].weight.data.fill_(0)
                self.ts_attn[i].bias.data.fill_(0.5)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, dim_out, global_pool=''):
        self.dim_out = dim_out
        self.head = nn.Linear(self.dim_feat, dim_out) if dim_out > 0 else nn.Identity()

    def forward(self, x, return_rep=True):
        B, F, J, C = x.shape
        x = x.reshape(-1, J, C)
        BF = x.shape[0]
        x = self.joints_embed(x)
        x = x + self.pos_embed
        _, J, C = x.shape
        x = x.reshape(-1, F, J, C) + self.temp_embed[:, :F, :, :]
        x = x.reshape(BF, J, C)
        x = self.pos_drop(x)
        alphas = []
        for idx, (blk_st, blk_ts) in enumerate(zip(self.blocks_st, self.blocks_ts)):
            x_st = blk_st(x, F)
            x_ts = blk_ts(x, F)
            if self.att_fuse:
                att = self.ts_attn[idx]
                alpha = torch.cat([x_st, x_ts], dim=-1)
                BF, J = alpha.shape[:2]
                alpha = att(alpha)
                alpha = alpha.softmax(dim=-1)
                x = x_st * alpha[:, :, 0:1] + x_ts * alpha[:, :, 1:2]
            else:
                x = (x_st + x_ts) * 0.5
        x = self.norm(x)
        x = x.reshape(B, F, J, -1)
        x = self.pre_logits(x)  # [B, F, J, dim_feat]
        if return_rep:
            return x
        x = self.head(x)
        return x

    def get_representation(self, x):
        return self.forward(x, return_rep=True)
