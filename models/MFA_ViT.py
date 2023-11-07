import torch
import torch.nn as nn
import torch.nn.functional as F
import collections.abc
import numpy as np
import math
import rtdl
from functools import partial
from itertools import repeat
from dataclasses import dataclass
from typing import Optional, Literal
from torch import Tensor
from torch.nn import Parameter

try:
    from models.regularizer import DropPath
except:
    from regularizer import DropPath

try:
    from torch import _assert
except ImportError:
    def _assert(condition: bool, message: str):
        assert condition, message


def cos_sin(x: Tensor) -> Tensor:
    """
        We borrow the code from the following link. Not all technical details are given in this documentation.

        Paper: https://arxiv.org/abs/2012.08986v2
        Code: https://github.com/mindspore-ai/models/tree/bdf2d8bcf11fe28e4ad3060cf2ddc818eacd8597/research/recommend/autodis
    """
    return torch.cat([torch.cos(x), torch.sin(x)], -1)


@dataclass
class AutoDisOptions:
    """
        We borrow the code from the following link. Not all technical details are given in this documentation.

        Paper: https://arxiv.org/abs/2012.08986v2
        Code: https://github.com/mindspore-ai/models/tree/bdf2d8bcf11fe28e4ad3060cf2ddc818eacd8597/research/recommend/autodis
    """
    n_meta_embeddings: int
    temperature: float


@dataclass
class PeriodicOptions:
    """
        We borrow the code from the following link. Not all technical details are given in this documentation.

        Paper: https://arxiv.org/abs/2012.08986v2
        Code: https://github.com/mindspore-ai/models/tree/bdf2d8bcf11fe28e4ad3060cf2ddc818eacd8597/research/recommend/autodis
    """
    n: int  # the output size is 2 * n
    sigma: float
    trainable: bool
    initialization: Literal['log-linear', 'normal']


class Periodic(nn.Module):
    """
        We borrow the code from the following link. Not all technical details are given in this documentation.

        Paper: https://arxiv.org/abs/2012.08986v2
        Code: https://github.com/mindspore-ai/models/tree/bdf2d8bcf11fe28e4ad3060cf2ddc818eacd8597/research/recommend/autodis
    """
    def __init__(self, n_features: int, options: PeriodicOptions) -> None:
        super().__init__()
        if options.initialization == 'log-linear':
            coefficients = options.sigma ** (torch.arange(options.n) / options.n)
            coefficients = coefficients[None].repeat(n_features, 1)
        else:
            assert options.initialization == 'normal'
            coefficients = torch.normal(0.0, options.sigma, (n_features, options.n))
        if options.trainable:
            self.coefficients = nn.Parameter(coefficients)  # type: ignore[code]
        else:
            self.register_buffer('coefficients', coefficients)

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 2
        return cos_sin(2 * torch.pi * self.coefficients[None] * x[..., None])


class NLinear(nn.Module):
    """
        We borrow the code from the following link. Not all technical details are given in this documentation.

        Paper: https://arxiv.org/abs/2012.08986v2
        Code: https://github.com/mindspore-ai/models/tree/bdf2d8bcf11fe28e4ad3060cf2ddc818eacd8597/research/recommend/autodis
    """
    def __init__(self, n: int, d_in: int, d_out: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = Parameter(Tensor(n, d_in, d_out))
        self.bias = Parameter(Tensor(n, d_out)) if bias else None
        with torch.no_grad():
            for i in range(n):
                layer = nn.Linear(d_in, d_out)
                self.weight[i] = layer.weight.T
                if self.bias is not None:
                    self.bias[i] = layer.bias

    def forward(self, x):
        assert x.ndim == 3
        x = x[..., None] * self.weight[None]
        x = x.sum(-2)
        if self.bias is not None:
            x = x + self.bias[None]
        return x


class NLinearMemoryEfficient(nn.Module):
    """
        We borrow the code from the following link. Not all technical details are given in this documentation.

        Paper: https://arxiv.org/abs/2012.08986v2
        Code: https://github.com/mindspore-ai/models/tree/bdf2d8bcf11fe28e4ad3060cf2ddc818eacd8597/research/recommend/autodis
    """
    def __init__(self, n: int, d_in: int, d_out: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for _ in range(n)])

    def forward(self, x):
        return torch.stack([l(x[:, i]) for i, l in enumerate(self.layers)], 1)


class NLayerNorm(nn.Module):
    """
        We borrow the code from the following link. Not all technical details are given in this documentation.

        Paper: https://arxiv.org/abs/2012.08986v2
        Code: https://github.com/mindspore-ai/models/tree/bdf2d8bcf11fe28e4ad3060cf2ddc818eacd8597/research/recommend/autodis
    """
    def __init__(self, n_features: int, d: int) -> None:
        super().__init__()
        self.weight = Parameter(torch.ones(n_features, d))
        self.bias = Parameter(torch.zeros(n_features, d))

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 3
        x = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True)
        x = self.weight * x + self.bias
        return x


class AutoDis(nn.Module):
    """
        We borrow the code from the link. Not all technical details are given in this documentation.

        Paper: https://arxiv.org/abs/2012.08986v2
        Code: https://github.com/mindspore-ai/models/tree/bdf2d8bcf11fe28e4ad3060cf2ddc818eacd8597/research/recommend/autodis
    """
    def __init__(
        self, n_features: int, d_embedding: int, options: AutoDisOptions
    ) -> None:
        super().__init__()
        self.first_layer = rtdl.NumericalFeatureTokenizer(
            n_features,
            options.n_meta_embeddings,
            False,
            'uniform',
        )
        self.leaky_relu = nn.LeakyReLU()
        self.second_layer = NLinear(
            n_features, options.n_meta_embeddings, options.n_meta_embeddings, False
        )
        self.softmax = nn.Softmax(-1)
        self.temperature = options.temperature
        # "meta embeddings" from the paper are just a linear layer
        self.third_layer = NLinear(
            n_features, options.n_meta_embeddings, d_embedding, False
        )
        # 0.01 is taken from the source code
        nn.init.uniform_(self.third_layer.weight, 0.01)

    def forward(self, x: Tensor) -> Tensor:
        x = self.first_layer(x)
        x = self.leaky_relu(x)
        x = self.second_layer(x)
        x = self.softmax(x / self.temperature)
        x = self.third_layer(x)
        return x


class AttributeEmbed(nn.Module):
    """
        1D Data to Embedding

        We borrow the code from the link. Not all technical details are given in this documentation.

        Paper: https://arxiv.org/abs/2012.08986v2
        Code: https://github.com/mindspore-ai/models/tree/bdf2d8bcf11fe28e4ad3060cf2ddc818eacd8597/research/recommend/autodis
    """
    def __init__(
        self,
        n_features: int,
        d_embedding: Optional[int],
        embedding_arch: None,
        periodic_options: Optional[PeriodicOptions],
        autodis_options: Optional[AutoDisOptions],
        d_feature: Optional[int],
        memory_efficient: bool,
        num_token: int,
    ) -> None:
        super().__init__()
        assert embedding_arch
        assert set(embedding_arch) <= {
            'linear',
            'positional',
            'relu',
            'shared_linear',
            'layernorm',
            'autodis',
        }
        if any(x in embedding_arch for x in ['linear', 'shared_linear', 'autodis']):
            assert d_embedding is not None
        else:
            assert d_embedding is None
        assert embedding_arch.count('positional') <= 1
        if 'autodis' in embedding_arch:
            assert embedding_arch == ['autodis']

        NLinear_ = NLinearMemoryEfficient if memory_efficient else NLinear
        layers: list[nn.Module] = []

        if embedding_arch[0] == 'linear':
            assert periodic_options is None
            assert autodis_options is None
            assert d_embedding is not None
            layers.append(
                rtdl.NumericalFeatureTokenizer(n_features, d_embedding, True, 'uniform')
                if d_feature is None
                else NLinear_(n_features, d_feature, d_embedding)
            )
            d_current = d_embedding
        elif embedding_arch[0] == 'positional':
            assert d_feature is None
            assert periodic_options is not None
            assert autodis_options is None
            layers.append(Periodic(n_features, periodic_options))
            d_current = periodic_options.n * 2
        elif embedding_arch[0] == 'autodis':
            assert d_feature is None
            assert periodic_options is None
            assert autodis_options is not None
            assert d_embedding is not None
            layers.append(AutoDis(n_features, d_embedding, autodis_options))
            d_current = d_embedding
        else:
            assert False

        for x in embedding_arch[1:]:
            layers.append(
                nn.ReLU()
                if x == 'relu'
                else NLinear_(n_features, d_current, d_embedding)  # type: ignore[code]
                if x == 'linear'
                else nn.Linear(d_current, d_embedding)  # type: ignore[code]
                if x == 'shared_linear'
                else NLayerNorm(n_features, d_current)  # type: ignore[code]
                if x == 'layernorm'
                else nn.Identity()
            )
            if x in ['linear', 'shared_linear']:
                d_current = d_embedding
            assert not isinstance(layers[-1], nn.Identity)
        self.d_embedding = d_current
        self.layers = nn.Sequential(*layers)
        self.proj = nn.Linear(n_features, num_token)

    def forward(self, x):
        x = self.layers(x) # BK -> BKC
        x = x.transpose(1, 2)  # BKC -> BCK
        x = self.proj(x)  # BCK -> BCN
        x = x.transpose(1, 2)  # BCN to BNC

        return x


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


# Define tuple
to_2tuple = _ntuple(2)


class PatchEmbed(nn.Module):
    """
        2D Image to Patch Embedding
    """
    def __init__(self, img_size=112, patch_size=8, view=1, in_chans=3, embed_dim=1024, norm_layer=None, flatten=True):
        """
            Configuration
            :param img_size: Size of input image, where default value is 112
            :param patch_size: Size of patch embeddings, where default value is 8
            :param view: Number of input images, where default value is 1 (face) or 2 (periocular)
            :param in_chans: Number of input channel, where default value is 3
            :param embed_dim: Size of token embeddings, where default value is 1024
            :param norm_layer: Norm layer, where default setting is None
            :param flatten: Flatten layer, where default vaue is True
        """
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(view * in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, V, C, H, W = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = x.reshape(B, V * C, H, W)  # shape => [B, N, H, W]
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BNHW -> BNC
        x = self.norm(x)

        return x


class Mlp(nn.Module):
    """
        Multi-layer Perceptron (MLP) Layer
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        """
            Configurations of MLP Layer.

            :param in_features: Size of input layer
            :param hidden_features: Size of hidden layer, where default value is None
            :param out_features: Size of output layer, where default value is None
            :param act_layer: Activation layer, where default setting is GELU
            :param drop: Dropout rate, where default value is 0.1
        """
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


class DWFC_MSA(nn.Module):
    """
        Depth-wise Fusion Convolutional-based Multi-head Self-Attention Layer (DWFC-MSA layer)
    """
    def __init__(self, dim, depth_block_channel, num_heads, attn_drop=0., proj_drop=0.):
        """
            Configurations of DWC-MSA Layer.

            :param dim: Dimension of token embeddings
            :param depth_block_channel: Size of input channel
            :param num_heads: Number of head
            :param attn_drop: Attention layer dropout rate, where default value is 0
            :param proj_drop: Dropout path rate, where default value is 0
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q, self.k, self.v = DWS_Conv(depth_block_channel), DWS_Conv(depth_block_channel), DWS_Conv(depth_block_channel)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        x = torch.reshape(x, (B, N, int(np.sqrt(C)), int(np.sqrt(C))))
        q, k, v = self.q(x), self.k(x), self.v(x)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MFA_layer(nn.Module):
    """
        Multimodal Fusion Attention Layer
    """
    def __init__(self, dim, depth_block_channel, num_heads, mlp_ratio=4., drop=0.1, attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """
            Configurations of MFA Layer.

            :param dim: Dimension of token embeddings
            :param num_heads: Number of head
            :param depth_block_channel: Size of input channel
            :param mlp_ratio: Ratio of mlp, where default value is 4
            :param drop: Dropout rate, where default value is 0.1
            :param attn_drop: Attention layer dropout rate, where default value is 0
            :param drop_path: Dropout path rate, where default value is 0
            :param act_layer: Activation layer, where default setting is GELU
            :param norm_layer: Norm layer, where default setting is LayerNorm
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = DWFC_MSA(dim, depth_block_channel=depth_block_channel, num_heads=num_heads,
                                   attn_drop=attn_drop, proj_drop=drop)
        # # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DWS_Conv(nn.Module):
    """
        Depth-wise Convolutional Layer (DWS-Conv)
    """
    def __init__(self, channels):
        """
            Configurations of Depth-wise Convolutional Layer.

            :param channels: Size of input channel
        """
        super(DWS_Conv, self).__init__()
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)

    def forward(self, x):
        out = self.depthwise(x)
        return out


class MFA_block(nn.Module):
    """
        Multimodal Fusion Attention Block
    """
    def __init__(self, dim, num_heads, depth_block_channel, mlp_ratio=4., drop=0.1,
                 attn_drop=0.1, drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """
            Configurations of MFA Block.

            :param dim: Dimension of token embeddings
            :param num_heads: Number of head
            :param depth_block_channel: Size of input channel
            :param mlp_ratio: Ratio of mlp, where default value is 4
            :param drop: Dropout rate, where default value is 0.1
            :param attn_drop: Attention layer dropout rate, where default value is 0
            :param drop_path: Dropout path rate, where default value is 0
            :param act_layer: Activation layer, where default setting is GELU
            :param norm_layer: Norm layer, where default setting is LayerNorm
        """
        super().__init__()
        self.transformer_block = MFA_layer(
            dim=dim, depth_block_channel=depth_block_channel, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer)
        self.depthwise_conv = DWS_Conv(channels=depth_block_channel)
        self.depthwise_conv2 = DWS_Conv(channels=depth_block_channel)
        self.conv_1x1 = nn.Conv2d(depth_block_channel * 2, depth_block_channel, kernel_size=1, padding=0)
        self.relu = nn.LeakyReLU()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        if len(x.shape) == 3:
            x_trans = self.transformer_block(x)

            B, C, F = x.shape
            x = torch.reshape(x, (B, C, int(np.sqrt(F)), int(np.sqrt(F))))
            x_conv = self.depthwise_conv(x)
            x_trans = torch.reshape(x_trans, (B, C, int(np.sqrt(F)), int(np.sqrt(F))))

            x = torch.reshape(x, (B, C, int(np.sqrt(F)), int(np.sqrt(F))))
            x0 = torch.cat((x_trans, x_conv), dim=1)
            x = x + self.drop(self.relu(self.conv_1x1(x0)))
        else:
            x_conv = self.depthwise_conv2(x)

            B, C, H, W = x.shape
            x = torch.reshape(x, (B, C, H * W))
            x_trans = self.transformer_block(x)
            x_trans = torch.reshape(x_trans, (B, C, H, W))

            x = torch.reshape(x, (B, C, H, W))
            x0 = torch.cat((x_trans, x_conv), dim=1)
            x = x + self.drop(self.relu(self.conv_1x1(x0)))

        return x


class MPT(nn.Module):
    """
        Multimodal-Prompt Tuning (MPT)
    """
    def __init__(self, channel):
        """
            Configurations of MPT.

            :param channel: Size of input channel
        """
        super().__init__()
        self.conv_1x1 = nn.Conv2d(channel * 2, channel, kernel_size=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv_1x1(x))
        return x


class MFA_ViT(nn.Module):
    """
        MFA-ViT Structure
    """
    def __init__(self, attr_size=47, img_size=112, patch_size=8, in_chans=3, embed_dim=1024, num_classes=9131,
                 layer_depth=4, num_heads=12, mlp_ratio=4., norm_layer=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., prompt_mode=None, prompt_tokens=32, head_strategy="prm"):
        """
            Configurations of MFA-ViT.

            :param attr_size: Size of soft-biometric attributes (1xN), where default value of N is 47
            :param img_size: Size of face modality, where default value is 112
            :param patch_size: Size of patch embedding, where default value is 8
            :param in_chans: Size of image channel, where default value is 3 (R,G,B)
            :param embed_dim: Dimension of token embeddings, where default value is 1024
            :param num_classes: Number of identities
            :param layer_depth: Number of layer depth in each block, where default value is 4
            :param num_heads: Number of head, where default value is 12
            :param mlp_ratio: Ratio of mlp, where default value is 4
            :param norm_layer: Norm layer, where default value is None
            :param drop_rate: Dropout rate, where default value is 0.1
            :param attn_drop_rate: Attention layer dropout rate, where default value is 0
            :param drop_path_rate: Dropout path rate, where default value is 0
            :param prompt_mode: Strategy of prompt, where default value is "deep"
            :param prompt_tokens: Size of prompt tokens, where default value is  32
            :param head_strategy: Type of classification token head, where default value is "prm"
        """
        super().__init__()

        self.layer_depth = layer_depth
        self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_patches = int(img_size / patch_size) ** 2
        self.num_classes = num_classes

        # Patch embeddings for all modalities: face, periocular, and attributes
        self.face_tokenizer = PatchEmbed(img_size=img_size, view=1, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.ocu_tokenizer = PatchEmbed(img_size=img_size, view=2, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.attr_tokenizer = AttributeEmbed(
            n_features=attr_size, d_embedding=embed_dim, embedding_arch=['positional', 'linear', 'relu'],
            periodic_options=PeriodicOptions(n=128, sigma=0.01,  trainable=False, initialization='log-linear',),
            autodis_options=None, d_feature=None, memory_efficient=True, num_token=(img_size//patch_size)**2
        )

        # Shared token embeddings for all modalities
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.resprompt_trans = MPT(channel=prompt_tokens)

        self.pos_drop = nn.Dropout(p=drop_rate)
        act_layer = nn.GELU

        # Prompt token embeddings
        self.prompt_mode = prompt_mode
        self.prompt_tokens = prompt_tokens
        embed_dim_sqrt = int(math.sqrt(embed_dim))
        if self.prompt_mode == "deep":
            self.prompts = nn.Parameter(torch.randn(layer_depth*2, 3, self.prompt_tokens, embed_dim_sqrt, embed_dim_sqrt))
        elif self.prompt_mode == "intermediate":
            self.prompts = nn.Parameter(torch.randn(2, 3, self.prompt_tokens, embed_dim_sqrt, embed_dim_sqrt))
        else:
            self.prompt_tokens = 0

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layer_depth)]

        self.conv_atten_block_1 = nn.Sequential(*[
            MFA_block(dim=self.embed_dim, num_heads=num_heads,
                      depth_block_channel=(self.num_patches + 1 + self.prompt_tokens), mlp_ratio=mlp_ratio,
                      drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], act_layer=act_layer,
                      norm_layer=norm_layer)
            for i in range(layer_depth)])
        self.conv_atten_block_2 = nn.Sequential(*[
            MFA_block(dim=self.embed_dim, num_heads=num_heads,
                      depth_block_channel=(self.num_patches + 1 + self.prompt_tokens), mlp_ratio=mlp_ratio,
                      drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], act_layer=act_layer,
                      norm_layer=norm_layer)
            for i in range(layer_depth)])

        # Classifier head
        self.head = nn.Linear(self.embed_dim, num_classes)
        self.head_strategy = head_strategy

    def tokenize(self, x, mode):
        """
            Embed the inputs into patch embeddings, including class and prompt embeddings.

            :param x: Input of modality
            :param mode:  Types of modality: "face", "ocular", and "attribute"
            :return: Patch embeddings
        """
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        if mode == "face" or mode == "f":
            x = self.face_tokenizer(x)
        elif mode == "periocular" or mode == "ocular" or mode == "ocu" or mode == "p":
            x = self.ocu_tokenizer(x)
        else:  # attr
            x = self.attr_tokenizer(x)

        x = torch.cat((cls_token, x), dim=1)
        if mode != "attribute" or mode != "a":
            x = self.pos_drop(x + self.pos_embed)

        return x

    def forward_features(self, x, mode):
        """
            Compute token embeddings based on the specified modality.

            :param x: Input of modality
            :param mode: Types of modality: "face", "ocular", and "attribute"
            :return: Feature embeddings
        """

        # Using deep prompt strategy
        if self.prompt_mode == "deep":
            if mode == "face" or mode == "f":
                prompt_tokens = torch.index_select(self.prompts, 1, torch.tensor((0), device=x.get_device()))
            elif mode == "periocular" or mode == "ocular" or mode == "ocu" or mode == "p":
                prompt_tokens = torch.index_select(self.prompts, 1, torch.tensor((1), device=x.get_device()))
            elif mode == "attribute" or mode == "a":
                prompt_tokens = torch.index_select(self.prompts, 1, torch.tensor((2), device=x.get_device()))

            # conv attention block 1
            for i in range(self.layer_depth):
                if i == 0:
                    x = torch.cat((prompt_tokens[i].reshape(1, self.prompt_tokens,
                                                            self.embed_dim).expand(x.shape[0], -1, -1), x), dim=1)
                else:
                    fusion = torch.cat((prompt_tokens[i-1].expand(x.shape[0], -1, -1, -1),
                                        prompt_tokens[i].expand(x.shape[0], -1, -1, -1)), dim=1)
                    resprompt = self.resprompt_trans(fusion)
                    x = torch.cat((resprompt, x), dim=1)
                x = self.conv_atten_block_1[i](x)

                if i == self.layer_depth-1: # last layer
                    x1 = x.clone()

                x = x[:, self.prompt_tokens:, :] # remove previous prompt

            # conv attention block 2
            for i in range(self.layer_depth):
                fusion = torch.cat((prompt_tokens[i-1+self.layer_depth].expand(x.shape[0], -1, -1, -1),
                                    prompt_tokens[i+self.layer_depth].expand(x.shape[0], -1, -1, -1)), dim=1)
                resprompt = self.resprompt_trans(fusion)
                x = torch.cat((resprompt, x), dim=1)
                x = self.conv_atten_block_2[i](x)

                if i == self.layer_depth-1: # last layer
                    x2 = x.clone()

                x = x[:, self.prompt_tokens:, :]  # remove previous prompt

        # Using intermediate prompt strategy
        elif self.prompt_mode == "intermediate":
            if mode == "face" or mode == "f":
                prompt_tokens = torch.index_select(self.prompts, 1, torch.tensor((0), device=x.get_device()))
            elif mode == "periocular" or mode == "ocular" or mode == "ocu" or mode == "p":
                prompt_tokens = torch.index_select(self.prompts, 1, torch.tensor((1), device=x.get_device()))
            elif mode == "attribute" or mode == "a":
                prompt_tokens = torch.index_select(self.prompts, 1, torch.tensor((2), device=x.get_device()))

            # conv attention block 1
            x = torch.cat((prompt_tokens[0].reshape(1, self.prompt_tokens,
                                                    self.embed_dim).expand(x.shape[0], -1, -1), x), dim=1)
            x = self.conv_atten_block_1(x)
            x1 = x.clone()

            # conv attention block 2
            x = x[:, self.prompt_tokens:, :] # remove previous prompt
            fusion = torch.cat((prompt_tokens[0].expand(x.shape[0], -1, -1, -1),
                                prompt_tokens[1].expand(x.shape[0], -1, -1, -1)), dim=1)
            resprompt = self.resprompt_trans(fusion)
            x = torch.cat((resprompt, x), dim=1)
            x2 = self.conv_atten_block_2(x)

        # Not using prompt strategy
        else:
            x1 = self.conv_atten_block_1(x)
            x2 = self.conv_atten_block_2(x1)

        # Addition operation between block 1 and block 2
        x = torch.add(x1, x2)

        B, N, H, W = x.shape
        x = torch.reshape(x, (B, N, -1))

        # Classfication head input type
        if self.prompt_mode is None:
            if self.head_strategy == "cls":
                x = x[:, 0, :]
            else:
                raise("head_strategy ({}) is not allowed for prompt_mode ({})".format(self.head_strategy,
                                                                                      self.prompt_mode))
        else:
            if self.head_strategy == "cls":
                x = x[:, self.prompt_tokens, :]
            elif self.head_strategy == "prm":
                x = x[:, 0:self.prompt_tokens, :].mean(dim=1)

        return x

    def forward(self, x_face, x_ocular, x_attr, return_feature=False):
        """
            Shared network structure by accepting face, periocular, soft-biometric attribures.

            :param x_face: Face patch embeddings
            :param x_ocular: Periocular patch embeddings
            :param x_attr: Soft-biometric attribute patch embeddings
            :param return_feature: Returning token embeddings and classification heads, where default value is False
            :return: Face and periocular classification heads
        """
        # Face modality
        x_face = self.tokenize(x_face, mode="face")
        x_face = self.forward_features(x_face, mode="face")
        y_face = self.head(x_face)

        # Periocular modality
        x_ocular = self.tokenize(x_ocular, mode="ocular")
        x_ocular = self.forward_features(x_ocular, mode="ocular")
        y_ocular = self.head(x_ocular)

        if return_feature:
            if x_attr is None:
                return x_face, x_ocular, y_face, y_ocular

            # Soft-biometric modality
            x_attr = self.tokenize(x_attr, mode="attribute")
            x_attr = self.forward_features(x_attr, mode="attribute")
            return x_face, x_ocular, x_attr, y_face, y_ocular
        else:
            return y_face, y_ocular


def model_params(model):
    """
        Information for total parameters with the given model.

        :param model: Model tensor
        :return: Size of total parameters
    """
    p_p = 0
    for p in list(model.parameters()):
        n_n = 1
        for s in list(p.size()):
            n_n *= s
        p_p += n_n
    return p_p
