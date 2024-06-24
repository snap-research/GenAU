import json
import importlib
import yaml
import os
from pathlib import Path
from typing import Any, IO
from omegaconf import OmegaConf

class IncludeLoader(yaml.SafeLoader):
    """
    Class extending the YAML Loader to handle nested documents
    YAML Loader with `!include` constructor.
    From: https://gist.github.com/joshbode/569627ced3076931b02f
    """
    def __init__(self, stream: IO) -> None:
        """
        Initialise Loader
        """
        # Registers the current directory as the root directory
        self.root = os.path.curdir

        super().__init__(stream)

class BaseConfiguration:
    """
    Represents the configuration parameters for running the process
    """
    def __init__(self, path):
        """
        Initializes the configuration with contents from the specified file
        :param path: path to the configuration file in json format
        """
        with open(path, 'r') as f:
            yaml_object = yaml.load(f, IncludeLoader)

        # Loads the configuration file and converts it to a dictionary
        omegaconf_config = OmegaConf.create(yaml_object, flags={"allow_objects": True}) # Uses the experimental "allow_objects" flag to allow classes and functions to be stored directly in the configuration
        self.config = OmegaConf.to_container(omegaconf_config, resolve=True)

        # Checks the configuration
        self.check_config()

        # Creates the directory structure
        self.create_directory_structure()

    def get_config(self):
        return self.config

    def check_config(self):
        """
        Checks that the configuration is well-formed
        Raises an exception if it is not
        :return:
        """
        pass

    def create_directory_structure(self):
        """
        Creates the directory structure needed by the configuration
        Eg. logging/checkpoints/results directories
        :return:
        """
        pass


class Configuration(BaseConfiguration):
    """
    Represents the configuration parameters for running the process
    """
    def __init__(self, path):
        """
        Initializes the configuration with contents from the specified file
        :param path: path to the configuration file in json format
        """
        super().__init__(path)

    def create_directory_structure(self):
        """
        Creates the directory structure needed by the configuration
        Eg. logging/checkpoints/results directories
        :return:
        """
        if "logging" in self.config and "checkpoints_directory" in self.config["logging"]:
            Path(self.config["logging"]["checkpoints_directory"]).mkdir(parents=True, exist_ok=True)

def get_class_by_name(name):
    """
    Gets a class by its fully qualified name
    :param name: fully qualified class name eg "mypackage.mymodule.MyClass"
    :return: the requested class
    """
    splits = name.split('.')
    module_name = '.'.join(splits[:-1])
    class_name = splits[-1]
    loaded_module = importlib.import_module(module_name)
    loaded_class = getattr(loaded_module, class_name)

    return loaded_class

def construct_include(loader: IncludeLoader, node: yaml.Node) -> Any:
    """
    Manages inclusion of the file referenced at node
    """
    filename = os.path.abspath(os.path.join(loader.root, loader.construct_scalar(node)))
    extension = os.path.splitext(filename)[1].lstrip('.')

    with open(filename, 'r') as f:
        if extension in ('yaml', 'yml'):
            return yaml.load(f, IncludeLoader) # Check if nested documents are handled correctly
        elif extension in ('json', ):
            return json.load(f)
        else:
            return ''.join(f.readlines())

def construct_module(loader: IncludeLoader, node: yaml.Node) -> Any:
    """
    Manages inclusion of a referenced function into the file
    """
    function_module_name = loader.construct_scalar(node)
    function = get_class_by_name(function_module_name)
    return function



# Registers the loader
yaml.add_constructor('!include', construct_include, IncludeLoader)
yaml.add_constructor('!module', construct_module, IncludeLoader)