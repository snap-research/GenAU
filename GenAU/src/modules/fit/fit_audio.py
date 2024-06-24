# Author: Willi Menapace
# Email: willi.menapace@gmail.com
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
from typing import Any, Callable, Dict, List, Tuple

from ..initializers.initializers import IdentityInitializer
from .layers.helpful import PositionalEmbedding

from .layers.rin_layers import divisible_by, LearnedSinusoidalPosEmb, Sequential, LayerNorm, FeedForward, repeat, exists, rearrange
from .layers.fit_layers import FITBlockV5, FITBlockHierarchical
from ..data_representation.data_entries import DataEntries
from ..data_representation.data_entry import DataEntry


class FIT(nn.Module):
    def __init__(
        self,
        get_logger: Callable=lambda: None,
        **model_config: Dict,
    ):
        super().__init__()

        self.get_logger = get_logger


        fit_block_module = model_config.get("fit_block_module", FITBlockV5)
        self.is_hierarchical_fit_block = False
        if fit_block_module is FITBlockHierarchical:
            self.is_hierarchical_fit_block = True
        linear_layer_module = model_config.get("linear_layer_module", nn.Linear)
        layer_normalization_module = model_config.get("layer_normalization_module", LayerNorm)
        feed_forward_module = model_config.get("feed_forward_module", FeedForward)

        self.input_size = model_config["input_size"] # (length, height, width) dimension of the input in pixels
        patch_size = model_config["patch_size"] # (length, height, width) dimension of the patch in pixels
        self.group_size = model_config["group_size"] # (length, height, width) dimension of the group in patches
        
        patch_channels = model_config["patch_channels"]
        input_channels = model_config.get("input_channels", 3)
        fit_blocks_count = model_config["fit_blocks_count"] # The number of blocks
        local_layers_per_block = model_config["local_layers_per_block"] # The number of local layers operating on all input patches divided by group for each FIT block
        global_layers_per_block = model_config["global_layers_per_block"] # The number of global layers operating on all latents for each FIT block
        latent_channels = model_config["latent_channels"]
        self.latent_count = model_config["latent_count"]
        self.time_pe_type = model_config.get("time_pe_type", "learned") # Type of positional encoding for the time information, either "learned" or "sinusoidal"
        learned_sinusoidal_pe_channels = model_config.get("learned_sinusoidal_dim", 16)
        
        self_conditioning_ff_config = model_config["self_conditioning_ff_config"]
        fit_block_config = model_config["fit_block_config"]

        # Retrieves the configuration for the nested latents if the FIT is hierarchical
        if self.is_hierarchical_fit_block:
            self.nested_dim_latent = fit_block_config["nested_latent_channels"]
            self.nested_latent_count = fit_block_config["nested_latent_count"]

        # Whether to augment the input with the average inputs
        self.extra_average_inputs = model_config.get("extra_average_inputs", {})
        # Whether to mask the input that is not extra input. Useful for debug purposes.
        self.mask_non_extra_input = model_config.get("mask_non_extra_input", False)

        self.time_scaling = model_config.get("time_scaling", 1.0)  # Rescales the time by this factor

        self.classes_count = model_config.get("classes_count", 0)  # Number of class labels, 0 = unconditional.

        # Parameters for context conditioning
        self.context_channels = model_config.get("context_channels", 0)  # Number of channels for context input
        self.summary_text_embeddings_channels = model_config.get("summary_text_embeddings_channels", 0)  # Number of channels for summary text embeddings input
        self.summary_text_embeddings_use_mask_token = model_config.get("summary_text_embeddings_use_mask_token", False)  # Whether to use a no mask token in the summary text embeddings
        self.conditioning_in_context = model_config.get("conditioning_in_context", False)  # Whether to add the conditioning information to the context

        # Whether to introduce a label indicating if the current media is an image or a audio
        self.use_audio_image_conditioning = model_config.get("use_audio_image_conditioning", False)
        # Whether to introduce information indicating the dataset from which the current sample was drawn
        self.dataset_ids_count = model_config.get("dataset_ids_count", 32) # The maximum dataset_id + 1 for embedding
        self.use_dataset_id_conditioning = model_config.get("use_dataset_id_conditioning", False)
        # Whether to introduce information indicating the original resolution of the current sample
        self.use_resolution_conditioning = model_config.get("use_resolution_conditioning", False)

        # Initializer for the network weights
        self.weight_initializer = IdentityInitializer({})
        if "weight_initializer" in model_config:
            target = model_config["weight_initializer"]["target"]
            self.weight_initializer = target(model_config["weight_initializer"])

        # Computes the dimensionality of the patches and groups
        self.image_size = self.input_size
        self.patch_size = patch_size
        self.patches_count, self.groups_count = self.compute_patch_and_groups_count(self.input_size)
        self.patches_count_total = self.patches_count[0] * self.patches_count[1] * self.patches_count[2]
        self.group_patches_total = self.group_size[0] * self.group_size[1] * self.group_size[2]
        self.groups_count_total = self.groups_count[0] * self.groups_count[1] * self.groups_count[2]

        # Computes the number of latents per group
        if self.latent_count % self.groups_count_total != 0:
            raise ValueError("The number of latents {} must be a multiple of the number of groups {}".format(self.latent_count, self.groups_count_total))
        self.latents_per_group = self.latent_count // self.groups_count_total

        self.input_channels = input_channels

        # Conditioning
        hidden_conditioning_features_count = patch_channels * 4

        # The number of latent tokens that are added to model conditioning signals
        self.latent_tokens_conditioning_count = 1 # Time is always added
        if self.classes_count > 0:
            self.latent_tokens_conditioning_count += 1

        # Time conditioning
        conditioning_features = latent_channels
        if self.conditioning_in_context:
            conditioning_features = self.context_channels
        if self.time_pe_type == "learned":
            time_fourier_features_channels = learned_sinusoidal_pe_channels + 1
            self.time_mlp = nn.Sequential(
                LearnedSinusoidalPosEmb(learned_sinusoidal_pe_channels),
                linear_layer_module(time_fourier_features_channels, hidden_conditioning_features_count),
                nn.GELU(),
                linear_layer_module(hidden_conditioning_features_count, conditioning_features)
            )                
        elif self.time_pe_type == "sinusoidal_pix2seq":
            self.time_mlp = nn.Sequential(
                linear_layer_module(conditioning_features // 4, conditioning_features),
                nn.SiLU(),
                linear_layer_module(conditioning_features, conditioning_features),
                nn.SiLU(),
            )
        else:
            raise Exception("Unknown time positional encoding type {}".format(self.time_pe_type))
        if self.use_audio_image_conditioning:
            self.audio_image_conditioning_mlp = nn.Sequential(
                linear_layer_module(2, conditioning_features) # 2 Features, image or audio
            )
            
        if self.use_dataset_id_conditioning:
            self.dataset_id_conditioning_mlp = nn.Sequential(
                linear_layer_module(self.dataset_ids_count, conditioning_features)
            )
        if self.use_resolution_conditioning:
            self.resolution_conditioning_mlp = nn.Sequential(
                linear_layer_module(2, hidden_conditioning_features_count),
                nn.GELU(),
                linear_layer_module(hidden_conditioning_features_count, conditioning_features)
            )

        # Class conditioning
        if self.classes_count > 0:
            self.classes_mlp = nn.Sequential(
                linear_layer_module(self.classes_count, hidden_conditioning_features_count),
                nn.GELU(),
                linear_layer_module(hidden_conditioning_features_count, conditioning_features)
            )    

        # Context mapping
        if self.context_channels:
            if self.summary_text_embeddings_channels:
                self.context_projection_summary_text_embeddings = linear_layer_module(in_features=self.summary_text_embeddings_channels, out_features=self.context_channels, bias=False)
            if self.summary_text_embeddings_use_mask_token:
                self.summary_text_embeddings_mask_token = nn.Parameter(torch.zeros((1, self.context_channels)))

        # pixels to patch and back
        pixels_in_patch = patch_size[0] * patch_size[1] * patch_size[2]
        
        # Checks if additional input channels are present
        patch_projection_input_channels = input_channels
        patch_projection_input_channels = patch_projection_input_channels + len(self.extra_average_inputs) * input_channels # + input_channels * self.use_lowres

        patch_projection_input_features = patch_projection_input_channels * pixels_in_patch
        self.to_patches = Sequential(
            self.weight_initializer(linear_layer_module(patch_projection_input_features, patch_channels), "to_patches"),
            nn.LayerNorm(patch_channels),
        )
        
        self.output_conv = model_config.get("output_conv", None)
        if self.output_conv:
            ouput_conv_channels = self.output_conv.get("channels", 16)
            ouput_conv_kernel_size = self.output_conv.get("kernel_size", (1, 7, 7))
            output_conv_padding = self.output_conv.get("padding", (0, 3, 3))
            output_conv_padding_mode = self.output_conv.get("padding_mode", "reflect")
            id_init = self.output_conv.get("id_init", True)

            self.output_conv = nn.Conv3d(ouput_conv_channels, input_channels,
                                         kernel_size=ouput_conv_kernel_size, padding=output_conv_padding,
                                         padding_mode=output_conv_padding_mode, bias=True)
            
            if id_init:
                self.output_conv.weight.data.zero_()
                self.output_conv.bias.data.zero_()
                self.output_conv.weight.data[0, 0, ouput_conv_kernel_size[0] // 2, ouput_conv_kernel_size[1] // 2, ouput_conv_kernel_size[2] // 2] = 1.0
                self.output_conv.weight.data[1, 1, ouput_conv_kernel_size[0] // 2, ouput_conv_kernel_size[1] // 2, ouput_conv_kernel_size[2] // 2] = 1.0
                self.output_conv.weight.data[2, 2, ouput_conv_kernel_size[0] // 2, ouput_conv_kernel_size[1] // 2, ouput_conv_kernel_size[2] // 2] = 1.0
        else:
            ouput_conv_channels = input_channels

        self.input_conv = model_config.get("input_conv", None)

        if self.input_conv is not None:       
            input_conv_kernel_size = self.input_conv.get("kernel_size", (1, 7, 7))
            stride = self.input_conv.get("stride", (1, 4, 4))
            input_conv_padding = self.input_conv.get("padding", (0, 3, 3))
            input_conv_padding_mode = self.input_conv.get("padding_mode", "reflect")
            
            self.input_conv = nn.Conv3d(patch_projection_input_channels, patch_channels, stride=stride,
                                        kernel_size=input_conv_kernel_size, padding=input_conv_padding,
                                        padding_mode=input_conv_padding_mode, bias=True)
        else:
            ouput_conv_channels = input_channels

        self.patch_to_pixels = nn.Sequential(
            layer_normalization_module(patch_channels),
            self.weight_initializer(linear_layer_module(patch_channels, ouput_conv_channels * pixels_in_patch), "to_pixels"),

        )

        self.latents = nn.Parameter(torch.randn(self.latent_count, latent_channels))
        nn.init.normal_(self.latents, std = 0.02)

        self.patches_pe = nn.Parameter(torch.randn([self.patches_count_total] + [patch_channels]))
        nn.init.normal_(self.patches_pe, std = 0.02)

        # Instantiates the latents for the hierarchy if needed
        if self.is_hierarchical_fit_block:
            self.nested_latents = nn.Parameter(torch.randn(self.nested_latent_count, self.nested_dim_latent))
            nn.init.normal_(self.nested_latents, std = 0.02)

        # The transformation that the self conditioning latents undergo before being summed to the latents
        self.self_conditioning_latents_projection = nn.Sequential(
            feed_forward_module(latent_channels, weight_initializer=self.weight_initializer, **self_conditioning_ff_config),
            layer_normalization_module(latent_channels)
        )

        if self.is_hierarchical_fit_block:
            self.nested_self_conditioning_latents_projection = nn.Sequential(
                feed_forward_module(self.nested_dim_latent, weight_initializer=self.weight_initializer, **self_conditioning_ff_config),
                layer_normalization_module(self.nested_dim_latent)
            )

        if hasattr(self.self_conditioning_latents_projection[-1], "gamma"): # The implementation of the LN can vary, so first check gamma is present
            nn.init.zeros_(self.self_conditioning_latents_projection[-1].gamma) # Beta is already to 0 by default

        # Instantiates the FIT block
        fit_blocks = []
        for idx in range(fit_blocks_count):
            fit_blocks.append(fit_block_module(patch_channels, dim_latent=latent_channels, dim_latent_conditioning=latent_channels, dim_context=self.context_channels, groups_count=self.groups_count, group_patches=self.group_size, group_patches_total=self.group_patches_total, groups_count_total=self.groups_count_total, latents_per_group=self.latents_per_group, local_layers_count=local_layers_per_block, global_layers_count=global_layers_per_block, weight_initializer=self.weight_initializer, block_config=fit_block_config))
        # The last layer comprises just local layers
        if local_layers_per_block > 0:
            fit_blocks.append(fit_block_module(patch_channels, dim_latent=latent_channels, dim_latent_conditioning=latent_channels, dim_context=self.context_channels, groups_count=self.groups_count, group_patches=self.group_size, group_patches_total=self.group_patches_total, groups_count_total=self.groups_count_total, latents_per_group=self.latents_per_group, local_layers_count=local_layers_per_block, global_layers_count=0, weight_initializer=self.weight_initializer, block_config=fit_block_config))
        self.blocks = nn.ModuleList(fit_blocks)

    @property
    def device(self):
        return next(self.parameters()).device

    def zero_init_linear(self, linear: nn.Module):
        """
        Attempts to zero initialize a linear layer of unknown type
        """
        if isinstance(linear, nn.Linear):
            if hasattr(linear, "weight"):
                nn.init.zeros_(linear.weight)
            if hasattr(linear, "bias"):
                nn.init.zeros_(linear.bias)
        else:
            raise Exception("Unknown way to zero init a layer of type {}".format(type(linear)))

    def compute_patch_and_groups_count(self, input_size: List[int]) -> Tuple[List[int]]:
        """
        Computes the number of patches that the input will be divided into.
        :param input_size: list with (..., length, height, width) shape of the input
        :return tuple with (length, height, width) number of patches
                (length, height, width) number of groups
        """

        input_size = input_size[-3:]

        patches_count = []
        for isize, psize in zip(input_size, self.patch_size):
            if not divisible_by(isize, psize):
                raise ValueError(f"All dimensions in size {input_size} must be divisible by patch size {self.patch_size}")
            patches_count.append(isize // psize)

        groups_count = []
        for pcount, gsize in zip(patches_count, self.group_size):
            if not divisible_by(pcount, gsize):
                raise ValueError(f"All dimensions in patch count {patches_count} must be divisible by group size {self.group_size}")
            groups_count.append(pcount // gsize)

        return patches_count, groups_count

    def trainer_data_to_audio(self, x: torch.Tensor):
        """
        :param x: (batch_size, channels, height, width) image input or
                  (batch_size, frames_count, channels, height, width) audio input.

        :return x (batch_size-size, channels, frames_count, height, width) audio input
        """

        shape_descriptor = {}

        # Makes the image a audio with a number of frames equal to the dimension of the time patch
        if len(x.shape) == 3: # B x C x T
            x = x.unsqueeze(-1) # B x C x T x F=1
            x = x.unsqueeze(2) # B x C x L x T x F=1
            x = x.repeat(1, 1, self.input_size[0], 1, 1)
            shape_descriptor["type"] = "1d"
        elif len(x.shape) == 4: # B x C x H x W
            x = x.unsqueeze(2) # B x C x L x H x W
            x = x.repeat(1, 1, self.input_size[0], 1, 1)
            shape_descriptor["type"] = "image"
        elif len(x.shape) == 5:
            x = x.permute(0, 2, 1, 3, 4)
            shape_descriptor["type"] = "audio"
        else:
            raise ValueError("Unknown input type of shape {}".format(x.shape))

        return x, shape_descriptor

    def audio_to_trainer_data(self, x: torch.Tensor, shape_descriptor: dict):
        """
        :param x: (batch_size, channels, frames_count, height, width) audio input.
        :return tensor with the shape of the original input coming from the trainer
        """

        if shape_descriptor["type"] == "1d":
            x = x.mean(dim=2).mean(dim=-1)  # Pool the predictions over the fake temporal and frequency dimension patch
        elif shape_descriptor["type"] == "image":
            x = x.mean(dim=2)  # Pool the predictions over the fake temporal dimension patch
        elif shape_descriptor["type"] == "audio":
            x = x.permute(0, 2, 1, 3, 4)
        else:
            raise ValueError("Unknown input type of shape {}".format(x.shape))

        return x

    def build_extra_average_inputs(self, x: torch.Tensor):
        """
        Builds a tensor with the average of the input as configured in the model config
        :param x: (batch_size, channels, frames_count, height, width) audio input
        :return (batch_size, channels * extra_inputs_counts, frames_count, heigth, width) tensor with the extra inputs
        """

        all_extra_inputs = []

        for current_extra_input_configuration in self.extra_average_inputs:
            kernel_size = current_extra_input_configuration["kernel_size"]
            input_scaling = current_extra_input_configuration["input_scaling"]

            averaging_mode = current_extra_input_configuration.get("mode", "pool_interpolate")
            if averaging_mode == "pool_interpolate":
                current_extra_input = torch.nn.functional.avg_pool3d(x, kernel_size=kernel_size, stride=kernel_size, padding=0)
                current_extra_input = current_extra_input * input_scaling
                current_extra_input = torch.nn.functional.interpolate(current_extra_input, scale_factor=kernel_size, mode="nearest")
            elif averaging_mode == "pool":
                paddings = []
                start_indexes = []
                for k in kernel_size:
                    paddings.append(k // 2)
                    # If the kernel is even, padding will generate an additional pixel which needs to be removes at the beginning
                    if k % 2 == 0:
                        start_indexes.append(1)
                    else:
                        start_indexes.append(0)
                current_extra_input = torch.nn.functional.avg_pool3d(x, kernel_size=kernel_size, stride=[1, 1, 1], padding=paddings)
                current_extra_input = current_extra_input[:, :, start_indexes[0]:, start_indexes[1]:, start_indexes[2]:]
            else:
                raise ValueError("Unknown averaging mode {}".format(averaging_mode))
            all_extra_inputs.append(current_extra_input)

        # Concatenates all the extra inputs
        if len(all_extra_inputs) > 0:
            extra_inputs = torch.cat(all_extra_inputs, dim=1)
        else:
            extra_inputs = None
        return extra_inputs

    def get_denoised_input(self, x: torch.Tensor, data_entries: DataEntries) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts the denoised input and denoised input masks from the data entries
        """
        if "original_input" not in data_entries.keys():
            raise Exception("The use of denosied input conditioning is required, but no original input is present in the data entries.")
        denoised_x_data_entry = data_entries["original_input"]
        denoised_x = denoised_x_data_entry.data
        denoised_x_mask = denoised_x_data_entry.mask
        if denoised_x_mask is None:
            raise Exception("Denoised input is present, but does not have a mask. It looks like the complete sample is already generated. Did you forget to set the mask?")
        if torch.all(denoised_x_mask):
            raise Exception("Denoised input is present, but the mask is all True. It looks like the complete sample is already generated. Did you forget to set the mask?")
        # Makes the mask of the same shape as the input
        denoised_x_mask = denoised_x_mask.reshape(list(denoised_x_mask.shape) + [1] * (denoised_x.ndim - denoised_x_mask.ndim))
        denoised_x_mask = denoised_x_mask.expand(denoised_x.shape)
        denoised_x, _ = self.trainer_data_to_audio(denoised_x)
        denoised_x_mask, _ = self.trainer_data_to_audio(denoised_x_mask)
        # Sanity check
        if denoised_x.shape != x.shape:
            raise Exception("The shape of the denoised input {} does not match the shape of the input {}".format(denoised_x.shape, x.shape))
        if denoised_x_mask.shape != x.shape:
            raise Exception("The shape of the denoised input mask {} does not match the shape of the input {}".format(denoised_x_mask.shape, x.shape))

        return denoised_x, denoised_x_mask

    def forward(self, data_entries: DataEntries):
        """
        :param data_entries: the input to the model. Must contain the following keys
            - input: (batch_size, channels, height, width) image input or
              (batch_size, frames_count, channels, height, width) audio input. Also has tag "input"
            - latents: (batch_size, latents_count, latents_dim) latents returned as results from the previous iteration or None is previous latents are not available
            - noise_labels: (batch_size) scalar linked to the diffusion timestep
            - class_labels: (batch_size) class labels or None

        :return data entries with the same format as the input, with "input" substituted with the network prediction and
            "latents" (batch_size, latents_count, latents_dim) substituted with the latents returned by the network. Also had tag "self-conditioning"
        """
        x = data_entries["input"].data

        # The text embeddings of the summary text
        # (batch_size, summary_text_emb_length, summary_text_emb_channels)
        summary_text_embeddings_entry = data_entries.get("summary_text_embeddings", None)
        summary_text_embeddings = None
        # (batch_size, summary_text_emb_length) 
        summary_text_embeddings_mask = None
        if summary_text_embeddings_entry is not None:
            summary_text_embeddings = summary_text_embeddings_entry.data
            summary_text_embeddings_mask = summary_text_embeddings_entry.mask
        
        enable_autocast = x.dtype == torch.float16 or x.dtype == torch.bfloat16
        # Checks if type casting needs to be enabled inside the model
        with torch.autocast("cuda", enabled=enable_autocast, dtype=x.dtype, cache_enabled=False): # cache_enabled put to False, otherwise the nested autocast enabling - disabling caused by Lightning and EDM precondition for some reason prevent projection layers in attention computation from receiving gradient
        
            # (batch_size)
            time = data_entries["noise_labels"].data
            time = time / self.time_scaling

            # Gets the latent self conditioning if present
            latent_self_conditioning = None
            latent_self_conditioning_mask = None
            nested_latent_self_conditioning = None
            nested_latent_self_conditioning_mask = None
            if "latents" in data_entries.keys():
                latent_self_conditioning = data_entries["latents"].data
                latent_self_conditioning_mask = data_entries["latents"].mask
                if self.is_hierarchical_fit_block:
                    nested_latent_self_conditioning = data_entries["nested_latents"].data
                    nested_latent_self_conditioning_mask = data_entries["nested_latents"].mask

            batch_size = x.shape[0]
            # Reshapes to the (batch_size, channels, frames_count, height, width) format
            x, shape_descriptor = self.trainer_data_to_audio(x)
            # Builds extra inputs and concatenates them to the input if necessary
            extra_inputs = self.build_extra_average_inputs(x)
            if extra_inputs is not None:
                if self.mask_non_extra_input:
                    x = x * 0.0
                x = torch.cat([x, extra_inputs], dim=1)
            

            # Embeds the time information
            # (batch_size, time_embedding_channels)
            time_embeddings = self.time_mlp(time)


            if self.use_audio_image_conditioning:
                audio_image_labels = data_entries["audio_image_labels"].data
                # (batch_size, 2)
                audio_image_labels = torch.nn.functional.one_hot(audio_image_labels, num_classes=2).to(x.dtype)
                # Embeds the audio image conditioning
                # (batch_size, conditioning_channels)
                audio_image_conditioning_embeddings = self.audio_image_conditioning_mlp(audio_image_labels)
                
            if self.use_dataset_id_conditioning:
                # (batch_size)
                dataset_id_labels = data_entries["dataset_id"].data
                dataset_id_mask = data_entries["dataset_id"].mask

                # (batch_size, dataset_ids_count)
                dataset_id_labels = torch.nn.functional.one_hot(dataset_id_labels, num_classes=self.dataset_ids_count).to(x.dtype)
                if dataset_id_mask is not None:
                    dataset_id_mask = dataset_id_mask.unsqueeze(-1)
                    dataset_id_labels = dataset_id_labels * dataset_id_mask
                # (batch_size, conditioning_channels)
                dataset_id_conditioning_embeddings = self.dataset_id_conditioning_mlp(dataset_id_labels)
            if self.use_resolution_conditioning:
                # (batch_size, 2)
                resolution_labels = data_entries["resolution"].data
                resolution_mask = data_entries["resolution"].mask

                # (batch_size, conditioning_channels)
                resolution_conditioning_embeddings = self.resolution_conditioning_mlp(resolution_labels)
                if resolution_mask is not None:
                    resolution_mask = resolution_mask.unsqueeze(-1)
                    resolution_conditioning_embeddings = resolution_conditioning_embeddings * resolution_mask

            # Extracts the class labels
            # (batch_size)
            class_labels_entry = data_entries.get("class_labels", None)
            class_labels = None
            class_labels_mask = None
            if class_labels_entry is not None:
                class_labels = class_labels_entry.data
                # (batch_size)
                class_labels_mask = class_labels_entry.mask
            # Embeds the class labels
            # (batch_size, classes_count)
            if self.classes_count == 0:
                class_labels = None
            elif class_labels is None:
                class_labels = torch.zeros([batch_size, self.classes_count], device=x.device)
            else:
                class_labels = torch.nn.functional.one_hot(class_labels, num_classes=self.classes_count).to(torch.float32)
                class_labels = class_labels.to(torch.float32).reshape(-1, self.classes_count)
            # (batch_size, conditioning_channels)
            if self.classes_count > 0:
                masked_class_labels = class_labels
                # (batch_size, 1)
                class_labels_mask_unsqueezed = class_labels_mask.unsqueeze(-1)
                # Eliminates the information of the masked class labels
                masked_class_labels = masked_class_labels * class_labels_mask_unsqueezed
                class_embeddings = self.classes_mlp(masked_class_labels)

            # Builds the context
            context = []
            if self.summary_text_embeddings_channels:
                # (batch_size, summary_text_emb_length, context_channels)
                encoded_summary_text_embeddings = self.context_projection_summary_text_embeddings(summary_text_embeddings)
                # Applies masking if required
                if summary_text_embeddings_mask is not None:
                    summary_text_embeddings_mask_reshaped = summary_text_embeddings_mask.reshape([summary_text_embeddings_mask.shape[0]] + [1] * (encoded_summary_text_embeddings.ndim - 1))
                    encoded_summary_text_embeddings = encoded_summary_text_embeddings * summary_text_embeddings_mask_reshaped
                    if self.summary_text_embeddings_use_mask_token:
                        # (batch_size, 1, context_channels)
                        summary_text_embeddings_mask_token = self.summary_text_embeddings_mask_token.unsqueeze(0).repeat(batch_size, 1, 1)
                        encoded_summary_text_embeddings = encoded_summary_text_embeddings + summary_text_embeddings_mask_token * torch.logical_not(summary_text_embeddings_mask_reshaped)

                context.append(encoded_summary_text_embeddings)


            if self.conditioning_in_context:
                # (batch_size, 1, context_channels)
                # Moayed: disabling the unsqueeze to match the dims of the conditing for image
                time_embeddings = time_embeddings.unsqueeze(1)
                context.append(time_embeddings)
                if self.use_audio_image_conditioning:
                    audio_image_conditioning_embeddings = audio_image_conditioning_embeddings.unsqueeze(1)
                    context.append(audio_image_conditioning_embeddings)
                if self.use_dataset_id_conditioning:
                    dataset_id_conditioning_embeddings = dataset_id_conditioning_embeddings.unsqueeze(1)
                    context.append(dataset_id_conditioning_embeddings)
                if self.use_resolution_conditioning:
                    resolution_conditioning_embeddings = resolution_conditioning_embeddings.unsqueeze(1)
                    context.append(resolution_conditioning_embeddings)
                if self.classes_count > 0:
                    # (batch_size, 1, context_channels)
                    # class_embeddings = class_embeddings.unsqueeze(1)
                    context.append(class_embeddings)
                

            # If the context is not empty, concatenates all the different contexts
            if len(context) > 0:
                context = torch.cat(context, dim=1)
            else:
                context = None

            # Expands the latents to the batch size
            # (batch_size, latents_count, latent_channels)
            latents = repeat(self.latents, 'n d -> b n d', b=batch_size)
            if self.is_hierarchical_fit_block:
                nested_latents = repeat(self.nested_latents, 'n d -> b n d', b=batch_size)

            # Sums the self conditioning latents if passed
            if exists(latent_self_conditioning):
                projected_self_conditioning_latents = self.self_conditioning_latents_projection(latent_self_conditioning)
                # Masks the latents not to be used in self conditioning
                if latent_self_conditioning_mask is not None:
                    latent_self_conditioning_mask_unsqueezed = latent_self_conditioning_mask.reshape(list(latent_self_conditioning_mask.shape) + [1] * (projected_self_conditioning_latents.ndim - latent_self_conditioning_mask.ndim))
                    projected_self_conditioning_latents = projected_self_conditioning_latents * latent_self_conditioning_mask_unsqueezed
                latents = latents + projected_self_conditioning_latents

                if self.is_hierarchical_fit_block:
                    projected_nested_self_conditioning_latents = self.nested_self_conditioning_latents_projection(nested_latent_self_conditioning)
                    if nested_latent_self_conditioning_mask is not None:
                        nested_latent_self_conditioning_mask_unsqueezed = nested_latent_self_conditioning_mask.reshape(list(nested_latent_self_conditioning_mask.shape) + [1] * (projected_nested_self_conditioning_latents.ndim - nested_latent_self_conditioning_mask.ndim))
                        projected_nested_self_conditioning_latents = projected_nested_self_conditioning_latents * nested_latent_self_conditioning_mask_unsqueezed
                    nested_latents = nested_latents + projected_nested_self_conditioning_latents

            # If the time token is to be used as an additional latent, add it. Otherwise it will be used in scale-shift conditioning
            entries_to_concatenate = []
            if not self.conditioning_in_context:
                time_embeddings = rearrange(time_embeddings, 'b d -> b 1 d')
                entries_to_concatenate.append(time_embeddings)
            if not self.conditioning_in_context:
                if self.use_audio_image_conditioning:
                    audio_image_conditioning_embeddings = rearrange(audio_image_conditioning_embeddings, 'b d -> b 1 d')
                    entries_to_concatenate.append(audio_image_conditioning_embeddings)
                if self.use_dataset_id_conditioning:
                    dataset_id_conditioning_embeddings = rearrange(dataset_id_conditioning_embeddings, 'b d -> b 1 d')
                    entries_to_concatenate.append(dataset_id_conditioning_embeddings)
                if self.use_resolution_conditioning:
                    resolution_conditioning_embeddings = rearrange(resolution_conditioning_embeddings, 'b d -> b 1 d')
                    entries_to_concatenate.append(resolution_conditioning_embeddings)
                if self.classes_count > 0:
                    class_embeddings = rearrange(class_embeddings, 'b d -> b 1 d')
                    entries_to_concatenate.append(class_embeddings)

            # (batch_size, conditioning_tokens, latent_channels) or None
            if len(entries_to_concatenate) > 0:
                latent_conditioning = torch.cat(entries_to_concatenate, dim=-2)
            else:
                latent_conditioning = None

            # Projects the input to patches
            # (batch_size, patch_count, patch_channels)
            if self.input_conv:
                patches = self.input_conv(x)
                patches = einops.rearrange(patches, 'b c t h w -> b (t h w) c')
                # TODO add handling of denoised input conditioning. Need to add a convolution for the denoised inputs and implement masking
            else:
                to_patch_input = einops.rearrange(x, 'b c (t p1) (h p2) (w p3) -> b (t h w) (c p1 p2 p3)', t=self.patches_count[0], h=self.patches_count[1], w=self.patches_count[2], p1=self.patch_size[0], p2=self.patch_size[1], p3=self.patch_size[2])
                patches = self.to_patches(to_patch_input)

            # Reorders the patches so that patches of the same group are contiguous
            # NOTE: could be fused with the previous rearrange
            patches = einops.rearrange(patches, "b (tgid tgsize hgid hgsize wgid wgsize) c -> b (tgid hgid wgid tgsize hgsize wgsize) c", tgid=self.groups_count[0], hgid=self.groups_count[1], wgid=self.groups_count[2], tgsize=self.group_size[0], hgsize=self.group_size[1], wgsize=self.group_size[2])

            # Adds PEs to the patches
            trimmed_patches_pe = self.patches_pe[:patches.shape[1], :]
            patches = patches + trimmed_patches_pe

            # Logs activation statistics
            current_log_name = "before_blocks_"
            # Forwards through each block
            for block_idx, block in enumerate(self.blocks):

                # Handles both nested and non-nested blocks by passing or not passing the nested latents
                if not self.is_hierarchical_fit_block:
                    # (batch_size, patch_count, patch_channels), (batch_size, latents_count, latent_channels)
                    patches, latents = block(patches, latents, latent_conditioning, context=context)
                else:
                    patches, latents, nested_latents = block(patches, latents, nested_latents, latent_conditioning, context=context)
                # Logs activation statistics
                current_log_name = "block_{:2d}_".format(block_idx)

            # Reorders the patches to the original order
            # NOTE could be fused with the subsequent rearrange
            patches = einops.rearrange(patches, "b (tgid hgid wgid tgsize hgsize wgsize) c -> b (tgid tgsize hgid hgsize wgid wgsize) c", tgid=self.groups_count[0], hgid=self.groups_count[1], wgid=self.groups_count[2], tgsize=self.group_size[0], hgsize=self.group_size[1], wgsize=self.group_size[2])
            # Transforms the patches back to pixels
            # (batch_size, channels, frames_count, height, width)
            pixels = self.patch_to_pixels(patches)
            pixels = einops.rearrange(pixels, 'b (t h w) (c p1 p2 p3) -> b c (t p1) (h p2) (w p3)', p1=self.patch_size[0], p2=self.patch_size[1], p3=self.patch_size[2], t=self.patches_count[0], h=self.patches_count[1], w=self.patches_count[2])

            if self.output_conv is not None:
                pixels = self.output_conv(pixels)

            # Transforms back to the original input shape
            pixels = self.audio_to_trainer_data(pixels, shape_descriptor)

            # Creates an output with the same entries as the input, but with the "input" key replaced by the model prediction
            outupt_data_entries = data_entries.shallow_copy()
            outupt_data_entries["input"].data = pixels
            outupt_data_entries.add(DataEntry("latents", latents, mask=latent_self_conditioning_mask, type="self-conditioning")) # Preserves the mask in case it is needed in further passes

            # Adds the nested latents to the output if they are present
            if self.is_hierarchical_fit_block:
                outupt_data_entries.add(DataEntry("nested_latents", nested_latents, mask=nested_latent_self_conditioning_mask, type="self-conditioning"))

            return outupt_data_entries
