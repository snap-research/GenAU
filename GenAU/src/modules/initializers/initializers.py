# Author: Willi Menapace
# Email: willi.menapace@gmail.com

from typing import Dict
import torch.nn as nn

class Initializer:

    def __init__(self, initializer_config: Dict):
        self.initializer_config = initializer_config

    def __call__(self, layer: nn.Module, layer_type: str) -> nn.Module:
        """
        Initializes the given module
        :param layer: The module to be initialized
        :param layer_type: A string identifying the type of module
        :return: The initialized module
        """

        raise NotImplementedError()
    
class IdentityInitializer(Initializer):
    """
    An initializer that does nothing
    """

    def __init__(self, initializer_config: Dict):
        super().__init__(initializer_config)


    def __call__(self, layer: nn.Module, layer_type: str):
        """
        Initializes the given module
        :param layer: The module to be initialized
        :param layer_type: A string identifying the type of module
        """

        return layer

class RINWeightScalerInitializer(Initializer):
    """
    An initializer that scales weights of a RIN module by a given factor
    """

    def __init__(self, initializer_config: Dict):
        super().__init__(initializer_config)

        self.scale = initializer_config["scale"]

    def __call__(self, layer: nn.Module, layer_type: str):
        """
        Initializes the given module
        :param layer: The module to be initialized
        :param layer_type: A string identifying the type of module
        """

        if layer_type == "to_patches":
            if isinstance(layer, nn.Linear):
                layer.weight.data *= self.scale
                if layer.bias is not None:
                    layer.bias.data *= self.scale
            return layer
        
        elif layer_type == "to_pixels":
            if isinstance(layer, nn.Linear):
                layer.weight.data *= self.scale
                if layer.bias is not None:
                    layer.bias.data *= self.scale
            return layer
        
        # Scales linear layers used for attention and feedforward operations
        if isinstance(layer, nn.Linear):
            layer.weight.data *= self.scale
            if layer.bias is not None:
                layer.bias.data *= self.scale

            return layer
        
        raise Exception("Unknown layer to initialize")
        return layer