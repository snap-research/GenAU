# Author: Willi Menapace
# Email: willi.menapace@gmail.com
from typing import List
from einops import rearrange

import torch
from torch import nn
import torch.nn.functional as F

from .rin_layers import DropPath, Attention, FeedForward
from ...initializers.initializers import Initializer

class FITBlockV5(nn.Module):
    """
    A FIT block with context conditioning through an additional cross attention before reading patches
    """

    def __init__(
        self,
        dim: int,
        dim_latent: int,
        dim_latent_conditioning: int,
        dim_context: int,
        groups_count: List[int],
        group_patches: List[int],
        group_patches_total: int,
        groups_count_total: int,
        latents_per_group: int,
        local_layers_count: int,
        global_layers_count: int,
        weight_initializer: Initializer,
        block_config = {}
    ):
        """
        :param dim: the number of input and output dimensions
        :param dim_latent: the number of dimensions of the latents and of latent conditionings
        :param dim_latent_conditioning: the number of dimensions of the latent conditioning
        :param dim_context: the number of dimensions of the context information, eg. text embeddings
        :param groups_count: the total number of groups in (time, height, width)
        :param group_patches: the size of each group in patches in (time, height, width)
        :param group_patches_total: the number of patches in each group
        :param groups_count_total: the number of groups
        :param latents_per_group: the number of latents per group
        :param local_layers_count: the number of local layers per block
        :param global_layers_count: the number of global layers per block
        :param weight_initializer: the initializer for the weights
        :param block_config: dictionary with additional configuration for the block
        """

        super().__init__()
        self.dim = dim
        self.dim_latent = dim_latent
        self.dim_latent_conditioning = dim_latent_conditioning
        self.dim_context = dim_context
        self.groups_count = groups_count
        self.group_patches = group_patches
        self.group_patches_total = group_patches_total
        self.groups_count_total = groups_count_total
        self.latents_per_group = latents_per_group
        self.local_layers_count = local_layers_count
        self.global_layers_count = global_layers_count

        self.weight_initializer = weight_initializer

        # Gets the configurations for the attention and feedforward networks
        default_attention_config = block_config["default_attention_config"]
        read_attention_config = block_config.get("read_attention_config", default_attention_config)
        read_context_attention_config = block_config.get("read_context_attention_config", default_attention_config)
        read_latent_conditioning_attention_config = block_config.get("read_latent_conditioning_attention_config", default_attention_config)
        write_attention_config = block_config.get("write_attention_config", default_attention_config)
        local_attention_config = block_config.get("local_attention_config", default_attention_config)
        global_attention_config = block_config.get("global_attention_config", default_attention_config)
        default_ff_config = block_config.get("default_ff_config", {})
        read_ff_config = block_config.get("read_ff_config", default_ff_config)
        read_context_ff_config = block_config.get("read_context_ff_config", default_ff_config)
        read_latent_conditioning_ff_config = block_config.get("read_latent_conditioning_ff_config", default_ff_config)
        write_ff_config = block_config.get("write_ff_config", default_ff_config)
        local_ff_config = block_config.get("local_ff_config", default_ff_config)
        global_ff_config = block_config.get("global_ff_config", default_ff_config)

        # Configuration for the dropout
        drop_units = block_config["drop_units"]
        drop_path = block_config["drop_path"]

        # If True, uses a feedforward network between each cross attention. If False, uses an identity
        # Defaults to False in hte original FIT implementation
        use_cross_attention_feedforward = block_config["use_cross_attention_feedforward"]

        attention_class = block_config.get("attention_class", Attention)
        ff_class = block_config.get("ff_class", FeedForward)

        # No need to read without global layers
        if self.global_layers_count > 0:

            # Read
            self.latents_attend_to_patches = attention_class(dim_latent, weight_initializer=self.weight_initializer, dim_context = dim, norm = True, norm_context = True, **read_attention_config)
            self.latents_cross_attn_ff = ff_class(dim_latent, weight_initializer=self.weight_initializer, drop_units=0.0, **read_ff_config) if use_cross_attention_feedforward else nn.Identity()

            # Context read
            if self.dim_context is not None:
                self.latents_attend_to_context = attention_class(dim_latent, weight_initializer=self.weight_initializer, dim_context = dim_context, norm = True, norm_context = True, **read_context_attention_config)
                self.latents_context_attn_ff = ff_class(dim_latent, weight_initializer=self.weight_initializer, drop_units=0.0, **read_context_ff_config) if use_cross_attention_feedforward else nn.Identity()


            # Latent conditioning read
            #self.latents_attent_to_latent_conditioning = attention_class(dim_latent, weight_initializer=self.weight_initializer, dim_context = dim_latent_conditioning, norm = True, norm_context = True, **read_latent_conditioning_attention_config)
            #self.latents_latent_conditioning_attn_ff = ff_class(dim_latent, weight_initializer=self.weight_initializer, drop_units=0.0, **read_latent_conditioning_ff_config) if use_cross_attention_feedforward else nn.Identity()

        # No need for local computation if there are no local layers
        if self.local_layers_count > 0:
            # Local layers
            self.local_layers = nn.ModuleList([])
            for _ in range(self.local_layers_count):
                self.local_layers.append(nn.ModuleList([
                    attention_class(dim, weight_initializer=self.weight_initializer, norm = True, **local_attention_config),
                    ff_class(dim, weight_initializer=self.weight_initializer, drop_units=drop_units, **local_ff_config)
                ]))

        # No need for global computation if there are no global layers
        if self.global_layers_count > 0:
            # Global layers
            self.global_layers = nn.ModuleList([])
            for _ in range(self.global_layers_count):
                self.global_layers.append(nn.ModuleList([
                    attention_class(dim_latent, weight_initializer=self.weight_initializer, norm = True, **global_attention_config),
                    ff_class(dim_latent, weight_initializer=self.weight_initializer, drop_units=drop_units, **global_ff_config)
                ]))
        
        # We write only if there are global layers that did some work on the latents
        if self.global_layers_count > 0:
            # Write
            self.patches_attend_to_latents = attention_class(dim, weight_initializer=self.weight_initializer, dim_context = dim_latent, norm = True, norm_context = True, **write_attention_config)
            self.patches_cross_attn_ff = ff_class(dim, weight_initializer=self.weight_initializer, drop_units=0.0, **write_ff_config) if use_cross_attention_feedforward else nn.Identity()

        if drop_path > 0.0:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = lambda x: x

    def forward(self, patches: torch.Tensor, latents: torch.Tensor, conditioning_latents: torch.Tensor, context: torch.Tensor):
        """
        :param patches (batch_size, patch_count, patch_channels) tensor with patches where patch_count = (tgid hgid wgid tgsize hgsize wgsize)
        :param latents (batch_size, latent_count, latent_channels) tensor with latents
        :param conditioning_latents (batch_size, conditioning_latent_count, latent_conditioning_channels) tensor with latent conditioning information. E.g. diffusion time
        :param context (batch_size, context_channels) tensor with context information. None if context is not present
        """

        if conditioning_latents is not None:
            raise Exception("A FITBlockV5 is used, but conditioning latents are specified. This block only supports context. Please add all the conditioning to the context.")

        # Applies the local network
        group_patches = rearrange(patches, 'b (g p) c -> (b g) p c', g=self.groups_count_total, p=self.group_patches_total)
        # Applies local layers only if they are present
        if self.local_layers_count > 0:
            for attn, ff in self.local_layers:
                group_patches = self.drop_path(attn(group_patches)) + group_patches
                group_patches = self.drop_path(ff(group_patches)) + group_patches

        # Applies all layers related to global computation
        if self.global_layers_count > 0:

            # Latents extract information from the conditioning latents
            #latents = self.latents_attent_to_latent_conditioning(latents, conditioning_latents) + latents
            #latents = self.latents_latent_conditioning_attn_ff(latents) + latents

            # latents extract information from the context
            if self.dim_context is not None:
                latents = self.latents_attend_to_context(latents, context) + latents 
                latents = self.latents_context_attn_ff(latents) + latents

            group_latents = rearrange(latents, 'b (g l) c -> (b g) l c', g=self.groups_count_total, l=self.latents_per_group)
            # Each group of latents reads from the corresponding group of patches
            group_latents = self.latents_attend_to_patches(group_latents, group_patches) + group_latents
            group_latents = self.latents_cross_attn_ff(group_latents) + group_latents
            latents = rearrange(group_latents, '(b g) l c -> b (g l) c', g=self.groups_count_total, l=self.latents_per_group)

            # Applies global self attention to all latents
            for attn, ff in self.global_layers:
                latents = self.drop_path(attn(latents)) + latents
                latents = self.drop_path(ff(latents)) + latents

            # Each group of patches attends to the respective group of latents
            group_latents = rearrange(latents, 'b (g l) c -> (b g) l c', g=self.groups_count_total, l=self.latents_per_group)
            group_patches = self.patches_attend_to_latents(group_patches, group_latents) + group_patches
            group_patches = self.patches_cross_attn_ff(group_patches) + group_patches

        patches = rearrange(group_patches, '(b g) p c -> b (g p) c', g=self.groups_count_total, p=self.group_patches_total)

        return patches, latents
    
class FITBlockHierarchical(nn.Module):
    """
    A hierarchical FIT block that uses nested FIT blocks rather than self attention for the global layers
    """

    def __init__(
        self,
        dim: int,
        dim_latent: int,
        dim_latent_conditioning: int,
        dim_context: int,
        groups_count: List[int],
        group_patches: List[int],
        group_patches_total: int,
        groups_count_total: int,
        latents_per_group: int,
        local_layers_count: int,
        global_layers_count: int,
        weight_initializer: Initializer,
        block_config = {}
    ):
        """
        :param dim: the number of input and output dimensions
        :param dim_latent: the number of dimensions of the latents and of latent conditionings
        :param dim_latent_conditioning: the number of dimensions of the latent conditioning
        :param dim_context: the number of dimensions of the context information, eg. text embeddings
        :param groups_count: the total number of groups in (time, height, width)
        :param group_patches: the size of each group in patches in (time, height, width)
        :param group_patches_total: the number of patches in each group
        :param groups_count_total: the number of groups
        :param latents_per_group: the number of latents per group
        :param local_layers_count: the number of local layers per block
        :param global_layers_count: the number of global layers per block
        :param weight_initializer: the initializer for the weights
        :param block_config: dictionary with additional configuration for the block
        """

        super().__init__()
        self.dim = dim
        self.dim_latent = dim_latent
        self.dim_latent_conditioning = dim_latent_conditioning
        self.dim_context = dim_context
        self.groups_count = groups_count
        self.group_patches = group_patches
        self.group_patches_total = group_patches_total
        self.groups_count_total = groups_count_total
        self.latents_per_group = latents_per_group
        self.local_layers_count = local_layers_count
        self.global_layers_count = global_layers_count

        self.weight_initializer = weight_initializer

        # Gets the configuration for the nested block
        nested_block_config = block_config.get("nested_block_config", {})

        self.nested_fit_block_class = block_config["nested_fit_block_module"]

        # The number of channels for the nested latents
        self.nested_dim_latent = block_config["nested_latent_channels"]
        # Number of latents for the nested latents
        self.nested_latent_count = block_config["nested_latent_count"]
        # Number of nested global and local layers
        self.nested_local_layers_per_block = block_config["nested_local_layers_per_block"]
        self.nested_global_layers_per_block = block_config["nested_global_layers_per_block"]

        # The number of groups in the upper level that each group in the nested level will contain (time, height, width)
        self.nested_group_size = block_config["nested_group_size"]

        self.nested_groups_count = self.compute_nested_groups_count()
        # Each group is made by latents_per_group and attends nested_group_size other groups. Arbitrarily put latents_per_group in the last dimension
        self.nested_group_patches = [self.nested_group_size[0], self.nested_group_size[1], self.nested_group_size[2] * self.latents_per_group]
        self.nested_group_patches_total = self.nested_group_patches[0] * self.nested_group_patches[1] * self.nested_group_patches[2]
        self.nested_groups_count_total = self.nested_groups_count[0] * self.nested_groups_count[1] * self.nested_groups_count[2]
        
        if self.nested_latent_count % self.nested_groups_count_total != 0:
            raise ValueError(f"Nested latent count {self.nested_latent_count} must be a multiple of the number of nested groups {self.nested_groups_count_total}")
        self.nested_latents_per_group = self.nested_latent_count // self.nested_groups_count_total

        # Gets the configurations for the attention and feedforward networks
        default_attention_config = block_config["default_attention_config"]
        read_attention_config = block_config.get("read_attention_config", default_attention_config)
        read_context_attention_config = block_config.get("read_context_attention_config", default_attention_config)
        read_latent_conditioning_attention_config = block_config.get("read_latent_conditioning_attention_config", default_attention_config)
        write_attention_config = block_config.get("write_attention_config", default_attention_config)
        local_attention_config = block_config.get("local_attention_config", default_attention_config)
        
        default_ff_config = block_config.get("default_ff_config", {})
        read_ff_config = block_config.get("read_ff_config", default_ff_config)
        read_context_ff_config = block_config.get("read_context_ff_config", default_ff_config)
        read_latent_conditioning_ff_config = block_config.get("read_latent_conditioning_ff_config", default_ff_config)
        write_ff_config = block_config.get("write_ff_config", default_ff_config)
        local_ff_config = block_config.get("local_ff_config", default_ff_config)
        
        # Configuration for the dropout
        drop_units = block_config["drop_units"]
        drop_path = block_config["drop_path"]

        # If True, uses a feedforward network between each cross attention. If False, uses an identity
        # Defaults to False in hte original FIT implementation
        use_cross_attention_feedforward = block_config["use_cross_attention_feedforward"]

        attention_class = block_config.get("attention_class", Attention)
        ff_class = block_config.get("ff_class", FeedForward)


        # No need to read without global layers
        if self.global_layers_count > 0:
            # Context read
            if self.dim_context is not None:
                self.latents_attend_to_context = attention_class(dim_latent, weight_initializer=self.weight_initializer, dim_context = dim_context, norm = True, norm_context = True, **read_context_attention_config)
                self.latents_context_attn_ff = ff_class(dim_latent, weight_initializer=self.weight_initializer, drop_units=0.0, **read_context_ff_config) if use_cross_attention_feedforward else nn.Identity()

            # Read
            self.latents_attend_to_patches = attention_class(dim_latent, weight_initializer=self.weight_initializer, dim_context = dim, norm = True, norm_context = True, **read_attention_config)
            self.latents_cross_attn_ff = ff_class(dim_latent, weight_initializer=self.weight_initializer, drop_units=0.0, **read_ff_config) if use_cross_attention_feedforward else nn.Identity()

            # Latent conditioning read
            #self.latents_attent_to_latent_conditioning = attention_class(dim_latent, weight_initializer=self.weight_initializer, dim_context = dim_latent_conditioning, norm = True, norm_context = True, **read_latent_conditioning_attention_config)
            #self.latents_latent_conditioning_attn_ff = ff_class(dim_latent, weight_initializer=self.weight_initializer, drop_units=0.0, **read_latent_conditioning_ff_config) if use_cross_attention_feedforward else nn.Identity()

        # No need for local computation if there are no local layers
        if self.local_layers_count > 0:
            # Local layers
            self.local_layers = nn.ModuleList([])
            for _ in range(self.local_layers_count):
                self.local_layers.append(nn.ModuleList([
                    attention_class(dim, weight_initializer=self.weight_initializer, norm = True, **local_attention_config),
                    ff_class(dim, weight_initializer=self.weight_initializer, drop_units=drop_units, **local_ff_config)
                ]))

        # No need for global computation if there are no global layers
        if self.global_layers_count > 0:
            # Global layers
            self.global_layers = nn.ModuleList([])
            for _ in range(self.global_layers_count):

                self.global_layers.append(
                    self.nested_fit_block_class(dim_latent, self.nested_dim_latent, self.dim_latent_conditioning, self.dim_context, self.nested_groups_count, self.nested_group_patches, self.nested_group_patches_total, self.nested_groups_count_total, self.nested_latents_per_group, self.nested_local_layers_per_block, self.nested_global_layers_per_block, weight_initializer=self.weight_initializer, block_config = nested_block_config)
                )
        
        # We write only if there are global layers that did some work on the latents
        if self.global_layers_count > 0:
            # Write
            self.patches_attend_to_latents = attention_class(dim, weight_initializer=self.weight_initializer, dim_context = dim_latent, norm = True, norm_context = True, **write_attention_config)
            self.patches_cross_attn_ff = ff_class(dim, weight_initializer=self.weight_initializer, drop_units=0.0, **write_ff_config) if use_cross_attention_feedforward else nn.Identity()

        if drop_path > 0.0:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = lambda x: x

    def compute_nested_groups_count(self):
        """
        Computes the number of nested groups
        """

        nested_groups_count = []
        for current_groups_count, current_nested_group_size in zip(self.groups_count, self.nested_group_size):
            if current_groups_count % current_nested_group_size != 0:
                raise ValueError(f"Group size {current_nested_group_size} must be a divisor of the number of groups {current_groups_count}")
            nested_groups_count.append(current_groups_count // current_nested_group_size)
        return nested_groups_count

    def forward(self, patches: torch.Tensor, latents: torch.Tensor, nested_latents: torch.Tensor, conditioning_latents: torch.Tensor, context: torch.Tensor):
        """
        :param patches (batch_size, patch_count, patch_channels) tensor with patches where patch_count = (tgid hgid wgid tgsize hgsize wgsize)
        :param latents (batch_size, latent_count, latent_channels) tensor with latents
        :param nested_latents: (batch_size, nested_latent_count, nested_latent_channels) tensor with nested latents. None if nested latents are not present
        :param conditioning_latents (batch_size, conditioning_latent_count, latent_conditioning_channels) tensor with latent conditioning information. E.g. diffusion time
        :param context (batch_size, context_channels) tensor with context information. None if context is not present
        """

        if conditioning_latents is not None:
            raise Exception("FITBlockHierarchical requires conditioning_latents to be concatenated with the context.")

        # Applies the local network
        group_patches = rearrange(patches, 'b (g p) c -> (b g) p c', g=self.groups_count_total, p=self.group_patches_total)
        # Applies local layers only if they are present
        if self.local_layers_count > 0:
            for attn, ff in self.local_layers:
                group_patches = self.drop_path(attn(group_patches)) + group_patches
                group_patches = self.drop_path(ff(group_patches)) + group_patches

        # Applies all layers related to global computation
        if self.global_layers_count > 0:
            # latents extract information from the context
            if self.dim_context is not None:
                latents = self.latents_attend_to_context(latents, context) + latents 
                latents = self.latents_context_attn_ff(latents) + latents

            # Latents extract information from the conditioning latents
            #latents = self.latents_attent_to_latent_conditioning(latents, conditioning_latents) + latents
            #latents = self.latents_latent_conditioning_attn_ff(latents) + latents

            group_latents = rearrange(latents, 'b (g l) c -> (b g) l c', g=self.groups_count_total, l=self.latents_per_group)
            # Each group of latents reads from the corresponding group of patches
            group_latents = self.latents_attend_to_patches(group_latents, group_patches) + group_latents
            group_latents = self.latents_cross_attn_ff(group_latents) + group_latents
            latents = rearrange(group_latents, '(b g) l c -> b (g l) c', g=self.groups_count_total, l=self.latents_per_group)

            # Reshape the latents to form the groups that will be used by the nested block. Puts the latents corresponding to multiple (or all) of the original groups in the same group
            global_layer_latents = rearrange(latents, 'b (tgid tsize hgid hsize wgid wsize) c -> b (tgid hgid wgid tsize hsize wsize) c', tgid=self.nested_groups_count[0], hgid=self.nested_groups_count[1], wgid=self.nested_groups_count[2], tsize=self.nested_group_patches[0], hsize=self.nested_group_patches[1], wsize=self.nested_group_patches[2])
            # Applies global self attention to all latents
            for nested_block in self.global_layers:                                                                       
                global_layer_latents, nested_latents = nested_block(global_layer_latents, nested_latents, conditioning_latents, context) 
            latents = rearrange(global_layer_latents, 'b (tgid hgid wgid tsize hsize wsize) c -> b (tgid tsize hgid hsize wgid wsize) c', tgid=self.nested_groups_count[0], hgid=self.nested_groups_count[1], wgid=self.nested_groups_count[2], tsize=self.nested_group_patches[0], hsize=self.nested_group_patches[1], wsize=self.nested_group_patches[2])

            # Each group of patches attends to the respective group of latents
            group_latents = rearrange(latents, 'b (g l) c -> (b g) l c', g=self.groups_count_total, l=self.latents_per_group)
            group_patches = self.patches_attend_to_latents(group_patches, group_latents) + group_patches
            group_patches = self.patches_cross_attn_ff(group_patches) + group_patches

        patches = rearrange(group_patches, '(b g) p c -> b (g p) c', g=self.groups_count_total, p=self.group_patches_total)

        return patches, latents, nested_latents
