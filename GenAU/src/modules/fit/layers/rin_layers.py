# Author: Willi Menapace
# Email: willi.menapace@gmail.com
import math
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import xformers
import xformers.ops
from xformers.ops import memory_efficient_attention
import xformers.components
import xformers.components.feedforward
from xformers.components.attention import ScaledDotProduct
from xformers.components.activations import Activation

from ...initializers.initializers import Initializer


# helpers functions
def normalize_magnitude(tensor: torch.Tensor, dim: int, eps=1e-4) -> torch.Tensor:
    """
    Normalizes the magnitude of the input tensor so that the average of its squares equals 1
    :param tensor: (dim1, dim2, ..., dimn) tensor with the input tensor
    :param dim: dimension along which to normalize the magnitude
    """

    square_input = tensor ** 2
    magnitude = torch.sqrt(torch.mean(square_input, dim=dim, keepdim=True)) + eps

    result = tensor / magnitude
    return result



def exists(x):
    return x is not None

def identity(x):
    return x

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def divisible_by(numer, denom):
    return (numer % denom) == 0

def safe_div(numer, denom, eps = 1e-10):
    return numer / denom.clamp(min = eps)

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    num_sqrt = math.sqrt(num)
    return int(num_sqrt) == num_sqrt

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

def Sequential(*mods):
    return nn.Sequential(*filter(exists, mods))

class DropPath(nn.Module):
    """
    A dropout procedure that drops the entire batch element with a certain probability
    """

    def __init__(self, drop_rate=0.0):
        super().__init__()
        self.drop_rate = drop_rate
        if self.drop_rate < 0.0 or self.drop_rate > 1.0:
            raise ValueError(f"Dropout rate must be between 0.0 and 1.0, but got {self.drop_rate}")
        
    def forward(self, x):
        """
        x: (batch_size, ...) tensor
        """
        if self.training and self.drop_rate > 0.0:
            keep_prob = 1.0 - self.drop_rate
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            random_tensor = random_tensor / keep_prob
            output = random_tensor * x

        else:
            output = x
        return output

# use layernorm without bias, more stable

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# positional embeds

class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        weight_initializer: Initializer,
        heads = 4,
        dim_head = 32,
        norm = False,
        conditioning_channels = None
    ):
        super().__init__()
        self.weight_initializer = weight_initializer
        hidden_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.conditioning_projection = None

        if exists(conditioning_channels):
            self.conditioning_projection = nn.Sequential(
                nn.SiLU(),
                self.weight_initializer(nn.Linear(conditioning_channels, dim * 2), "attention"),
                Rearrange('b d -> b 1 d')
            )

            nn.init.zeros_(self.conditioning_projection[-2].weight)
            nn.init.zeros_(self.conditioning_projection[-2].bias)

        self.norm = LayerNorm(dim) if norm else nn.Identity()

        self.to_qkv = self.weight_initializer(nn.Linear(dim, hidden_dim * 3, bias = False), "attention")

        self.to_out = nn.Sequential(
            self.weight_initializer(nn.Linear(hidden_dim, dim, bias = False), "attention"),
            LayerNorm(dim)
        )

    def forward(
        self,
        x,
        conditioning_embeddings = None
    ):
        h = self.heads
        x = self.norm(x)

        if exists(self.conditioning_projection):
            assert exists(conditioning_embeddings)
            scale, shift = self.conditioning_projection(conditioning_embeddings).chunk(2, dim = -1)
            x = (x * (scale + 1)) + shift

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        context = torch.einsum('b h n d, b h n e -> b h d e', k, v)

        out = torch.einsum('b h d e, b h n d -> b h n e', context, q)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        weight_initializer: Initializer,
        dim_context = None,
        heads = 4,
        dim_head = 32,
        norm = False,
        norm_context = False,
        conditioning_channels = None
    ):
        super().__init__()
        self.weight_initializer = weight_initializer
        hidden_dim = dim_head * heads
        dim_context = default(dim_context, dim)

        self.conditioning_projection = None

        if exists(conditioning_channels):
            self.conditioning_projection = nn.Sequential(
                nn.SiLU(),
                self.weight_initializer(nn.Linear(conditioning_channels, dim * 2), "attention"),
                Rearrange('b d -> b 1 d')
            )

            nn.init.zeros_(self.conditioning_projection[-2].weight)
            nn.init.zeros_(self.conditioning_projection[-2].bias)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.norm = LayerNorm(dim) if norm else nn.Identity()
        self.norm_context = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.to_q = self.weight_initializer(nn.Linear(dim, hidden_dim, bias = False), "attention")
        self.to_kv = self.weight_initializer(nn.Linear(dim_context, hidden_dim * 2, bias = False), "attention")
        self.to_out = self.weight_initializer(nn.Linear(hidden_dim, dim, bias = False), "attention")

    def forward(
        self,
        x,
        context = None,
        conditioning_embeddings = None
    ):
        h = self.heads

        if exists(context):
            context = self.norm_context(context)

        x = self.norm(x)

        context = default(context, x)

        if exists(self.conditioning_projection):
            assert exists(conditioning_embeddings)
            scale, shift = self.conditioning_projection(conditioning_embeddings).chunk(2, dim = -1)
            x = (x * (scale + 1)) + shift

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        q = q * self.scale

        # Executes attention computation in FP32
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class CosineAttention(nn.Module):
    def __init__(
        self,
        dim,
        weight_initializer: Initializer,
        dim_context = None,
        heads = 4,
        dim_head = 32,
        eps = 1e-4,
        norm = False,
        norm_context = False,
        conditioning_channels = None
    ):
        super().__init__()
        self.weight_initializer = weight_initializer
        hidden_dim = dim_head * heads
        dim_context = default(dim_context, dim)
        self.eps = eps

        self.conditioning_projection = None

        if exists(conditioning_channels):
            self.conditioning_projection = nn.Sequential(
                nn.SiLU(),
                self.weight_initializer(nn.Linear(conditioning_channels, dim * 2), "attention"),
                Rearrange('b d -> b 1 d')
            )

            nn.init.zeros_(self.conditioning_projection[-2].weight)
            nn.init.zeros_(self.conditioning_projection[-2].bias)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.norm = LayerNorm(dim) if norm else nn.Identity()
        self.norm_context = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.to_q = self.weight_initializer(nn.Linear(dim, hidden_dim, bias = False), "attention")
        self.to_kv = self.weight_initializer(nn.Linear(dim_context, hidden_dim * 2, bias = False), "attention")
        self.to_out = self.weight_initializer(nn.Linear(hidden_dim, dim, bias = False), "attention")

    def forward(
        self,
        x,
        context = None,
        conditioning_embeddings = None
    ):
        h = self.heads

        if exists(context):
            context = self.norm_context(context)

        x = self.norm(x)

        context = default(context, x)

        if exists(self.conditioning_projection):
            assert exists(conditioning_embeddings)
            scale, shift = self.conditioning_projection(conditioning_embeddings).chunk(2, dim = -1)
            x = (x * (scale + 1)) + shift

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        q = normalize_magnitude(q, dim=-1, eps=self.eps)
        k = normalize_magnitude(k, dim=-1, eps=self.eps)

        # Executes attention computation in FP32
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim * self.scale
        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class XFormersEfficientAttention(nn.Module):
    def __init__(
        self,
        dim,
        weight_initializer: Initializer,
        dim_context = None,
        heads = 4,
        dim_head = 32,
        norm = False,
        norm_context = False,
        conditioning_channels = None,
        xformers_efficient_attention_fw = xformers.ops.fmha.cutlass.FwOp,
        xformers_efficient_attention_bw = xformers.ops.fmha.cutlass.BwOp,
    ):
        super().__init__()
        self.weight_initializer = weight_initializer
        self.xformers_efficient_attention_fw = xformers_efficient_attention_fw
        self.xformers_efficient_attention_bw = xformers_efficient_attention_bw

        if self.xformers_efficient_attention_fw == "flash":
            self.xformers_efficient_attention_fw = xformers.ops.fmha.flash.FwOp
        if self.xformers_efficient_attention_bw == "flash":
            self.xformers_efficient_attention_bw = xformers.ops.fmha.flash.BwOp

        hidden_dim = dim_head * heads
        dim_context = default(dim_context, dim)

        self.conditioning_projection = None

        if exists(conditioning_channels):
            self.conditioning_projection = nn.Sequential(
                nn.SiLU(),
                self.weight_initializer(nn.Linear(conditioning_channels, dim * 2), "attention"),
                Rearrange('b d -> b 1 d')
            )

            nn.init.zeros_(self.conditioning_projection[-2].weight)
            nn.init.zeros_(self.conditioning_projection[-2].bias)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.norm = LayerNorm(dim) if norm else nn.Identity()
        self.norm_context = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.to_q = self.weight_initializer(nn.Linear(dim, hidden_dim, bias = False), "attention")
        self.to_kv = self.weight_initializer(nn.Linear(dim_context, hidden_dim * 2, bias = False), "attention")
        self.to_out = self.weight_initializer(nn.Linear(hidden_dim, dim, bias = False), "attention")

    def forward(
        self,
        x,
        context = None,
        conditioning_embeddings = None
    ):
        h = self.heads

        if exists(context):
            context = self.norm_context(context)

        x = self.norm(x)

        context = default(context, x)

        if exists(self.conditioning_projection):
            assert exists(conditioning_embeddings)
            scale, shift = self.conditioning_projection(conditioning_embeddings).chunk(2, dim = -1)
            x = (x * (scale + 1)) + shift

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h = h), qkv)

        # Computes the attention
        out = memory_efficient_attention(q, k, v, op=(self.xformers_efficient_attention_fw, self.xformers_efficient_attention_bw))

        out = rearrange(out, 'b n h d -> b n (h d)')
        return self.to_out(out)

class XFormersAttention(nn.Module):
    def __init__(
        self,
        dim,
        weight_initializer: Initializer,
        dim_context = None,
        heads = 4,
        dim_head = 32,
        norm = False,
        norm_context = False,
        conditioning_channels = None,
        xformers_attention_class = ScaledDotProduct
    ):
        super().__init__()
        self.weight_initializer = weight_initializer # TODO check
        hidden_dim = dim_head * heads
        dim_context = default(dim_context, dim)

        self.conditioning_projection = None

        if exists(conditioning_channels):
            self.conditioning_projection = nn.Sequential(
                nn.SiLU(),
                self.weight_initializer(nn.Linear(conditioning_channels, dim * 2), "attention"),
                Rearrange('b d -> b 1 d')
            )

            nn.init.zeros_(self.conditioning_projection[-2].weight)
            nn.init.zeros_(self.conditioning_projection[-2].bias)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.norm = LayerNorm(dim) if norm else nn.Identity()
        self.norm_context = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.to_q = self.weight_initializer(nn.Linear(dim, hidden_dim, bias = False), "attention")
        self.to_kv = self.weight_initializer(nn.Linear(dim_context, hidden_dim * 2, bias = False), "attention")
        self.to_out = self.weight_initializer(nn.Linear(hidden_dim, dim, bias = False), "attention")

        self.xformer_attention = xformers_attention_class()

    def forward(
        self,
        x,
        context = None,
        conditioning_embeddings = None
    ):
        h = self.heads

        if exists(context):
            context = self.norm_context(context)

        x = self.norm(x)

        context = default(context, x)

        if exists(self.conditioning_projection):
            assert exists(conditioning_embeddings)
            scale, shift = self.conditioning_projection(conditioning_embeddings).chunk(2, dim = -1)
            x = (x * (scale + 1)) + shift

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # Computes the attention
        out = self.xformer_attention(q, k, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, weight_initializer: Initializer, mult = 4, conditioning_channels = None, drop_units = 0.0, use_layer_normalization=True):
        """
        :param dim: the number of input and output dimensions
        :param weight_initializer: the initializer for the weights
        :param mult: the multiplier for the hidden dimension
        :param conditioning_channels: the number of channles that the conditioning signal si insert through scale and shift contains
        :param drop_units: dropout probability of hidden activations
        :param use_layer_normalization: whether to use layer normalization
        """
        
        super().__init__()
        self.weight_initializer = weight_initializer

        self.use_layer_normalization = use_layer_normalization
        self.norm = nn.Identity()
        if self.use_layer_normalization:
            self.norm = LayerNorm(dim)

        self.conditioning_projection = None

        if exists(conditioning_channels):
            self.conditioning_projection = nn.Sequential(
                nn.SiLU(),
                self.weight_initializer(nn.Linear(conditioning_channels, dim * 2), "feed_forward"),
                Rearrange('b d -> b 1 d')
            )

            nn.init.zeros_(self.conditioning_projection[-2].weight)
            nn.init.zeros_(self.conditioning_projection[-2].bias)

        inner_dim = int(dim * mult)
        sequential_layers = [
            self.weight_initializer(nn.Linear(dim, inner_dim), "feed_forward"),
            nn.GELU(),
        ]
        if drop_units > 0.0:
            sequential_layers.append(nn.Dropout(drop_units))
        sequential_layers.append(self.weight_initializer(nn.Linear(inner_dim, dim), "feed_forward"))
        self.net = nn.Sequential(
            *sequential_layers
        )

    def forward(self, x, conditioning_embeddings = None):
        x = self.norm(x)

        if exists(self.conditioning_projection):
            assert exists(conditioning_embeddings)
            scale, shift = self.conditioning_projection(conditioning_embeddings).chunk(2, dim = -1)
            x = (x * (scale + 1)) + shift

        return self.net(x)


class XFormersFeedForward(nn.Module):
    def __init__(self, dim, weight_initializer: Initializer, mult = 4, conditioning_channels = None, drop_units = 0.0, use_layer_normalization=True, use_bias=True):
        """
        :param dim: the number of input and output dimensions
        :param weight_initializer: the initializer for the weights
        :param mult: the multiplier for the hidden dimension
        :param conditioning_channels: the number of channles that the conditioning signal si insert through scale and shift contains
        :param drop_units: dropout probability of hidden activations
        :param use_layer_normalization: whether to use layer normalization
        """
        
        super().__init__()
        self.weight_initializer = weight_initializer

        self.use_layer_normalization = use_layer_normalization
        self.norm = nn.Identity()
        if self.use_layer_normalization:
            self.norm = LayerNorm(dim)

        self.conditioning_projection = None

        if exists(conditioning_channels):
            raise Exception("Conditioning is not supported by XFormersFeedForward")

        self.net = xformers.components.feedforward.MLP(dim, drop_units, Activation.GeLU, mult, use_bias)

    def forward(self, x, conditioning_embeddings = None):
        x = self.norm(x)

        if exists(self.conditioning_projection):
            assert exists(conditioning_embeddings)
            raise Exception("Conditioning embeddings are not supported by XFormersFeedForward")

        return self.net(x)



# model
class RINBlock(nn.Module):
    def __init__(
        self,
        dim,
        latent_self_attn_depth,
        weight_initializer: Initializer,
        dim_latent = None,
        dim_context = None,
        attention_config = {},
        drop_units = 0.0,
        drop_path = 0.0,
        block_config = {}
    ):
        """
        :param dim: the number of input and output dimensions
        :param latent_self_attn_depth: the number of latent self-attention layers
        :param weight_initializer: the initializer for the weights
        :param dim_latent: the number of dimensions of the latents
        :param dim_context: the number of dimensions of the context information, eg. text embeddings
        :param attention_config: deprecated parameter for the attention configuration. Insert this information in block_config
        :param drop_units: probability to dropout hidden activations in the feedforward network of the latent self-attention
        :param drop_path: probability to drop the entire latent self-attention block
        :param block_config: dictionary with additional configuration for the block
        """
        
        super().__init__()
        dim_latent = default(dim_latent, dim)
        self.weight_initializer = weight_initializer

        # Gets the configurations for the attention and feedforward networks
        if "attention_config" in block_config:
            attention_config = block_config["attention_config"]
        read_attention_config = block_config.get("read_attention_config", attention_config)
        write_attention_config = block_config.get("write_attention_config", attention_config)
        ff_config = block_config.get("ff_config", {})
        read_ff_config = block_config.get("read_ff_config", ff_config)
        write_ff_config = block_config.get("write_ff_config", ff_config)

        self.latents_attend_to_patches = Attention(dim_latent, weight_initializer=self.weight_initializer, dim_context = dim, norm = True, norm_context = True, **read_attention_config)
        self.latents_cross_attn_ff = FeedForward(dim_latent, weight_initializer=self.weight_initializer, **read_ff_config)

        self.latent_self_attns = nn.ModuleList([])
        for _ in range(latent_self_attn_depth):
            self.latent_self_attns.append(nn.ModuleList([
                Attention(dim_latent, weight_initializer=self.weight_initializer, norm = True, **attention_config),
                FeedForward(dim_latent, weight_initializer=self.weight_initializer, drop_units = drop_units, **ff_config)
            ]))

        if drop_path > 0.0:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = lambda x: x

        self.patches_attend_to_latents = Attention(dim, weight_initializer=self.weight_initializer, dim_context = dim_latent, norm = True, norm_context = True, **write_attention_config)
        self.patches_cross_attn_ff = FeedForward(dim, weight_initializer=self.weight_initializer, **write_ff_config)

    def forward(self, patches: torch.Tensor, latents: torch.Tensor, conditioning_embeddings: torch.Tensor, context: torch.Tensor):
        """
        :param patches (batch_size, patch_count, patch_channels) tensor with patches
        :param latents (batch_size, latent_count, latent_channels) tensor with latents
        :param conditioning_embeddings (batch_size, conditioning_channels) tensor with conditioning embeddings
        :param context (batch_size, context_channels) tensor with context information. None if context is not present
        """

        # latents extract or cluster information from the patches
        latents = self.latents_attend_to_patches(latents, patches, conditioning_embeddings=conditioning_embeddings) + latents
        latents = self.latents_cross_attn_ff(latents, conditioning_embeddings=conditioning_embeddings) + latents

        # latent self attention
        for attn, ff in self.latent_self_attns:
            latents = self.drop_path(attn(latents, conditioning_embeddings=conditioning_embeddings)) + latents
            latents = self.drop_path(ff(latents, conditioning_embeddings=conditioning_embeddings)) + latents

        # patches attend to the latents
        patches = self.patches_attend_to_latents(patches, latents, conditioning_embeddings=conditioning_embeddings) + patches
        patches = self.patches_cross_attn_ff(patches, conditioning_embeddings=conditioning_embeddings) + patches

        return patches, latents
