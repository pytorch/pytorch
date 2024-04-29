# This file codify PT2 Inference Archive Spec
# https://docs.google.com/document/d/1jLPp8MN8Whs0-VW9PmJ93Yg02W85tpujvHrTa1pc5x8/edit?usp=sharing

# Naming convention
# *_DIR: path to a folder, e.g. "data/aotinductor/"
# *_PATH: absolute path to a file, e.g. "models/merge.json"
# *_FORMAT: naming format of a file, e.g. "models/{}.json"

ARCHIVE_ROOT_NAME: str = "package"

# Archive format
ARCHIVE_FORMAT_PATH: str = "archive_format"

# Model definitions
MODELS_DIR: str = "models/"
MODELS_FILENAME_FORMAT: str = "models/{}.json"  # {model_name}

# AOTInductor artifacts
AOTINDUCTOR_DIR: str = "data/aotinductor/"

# weights, including parameters and buffers
WEIGHTS_DIR: str = "data/weights/"
WEIGHT_FILENAME_PREFIX: str = "weight_"

# constants, including tensor_constants, non-persistent buffers and script objects
CONSTANTS_DIR: str = "data/constants/"
TENSOR_CONSTANT_FILENAME_PREFIX: str = "tensor_"
CUSTOM_OBJ_FILENAME_PREFIX: str = "custom_obj_"

# sample inputs
SAMPLE_INPUTS_DIR: str = "data/sample_inputs/"
SAMPLE_INPUTS_FILENAME_FORMAT: str = "data/sample_inputs/{}.pt"  # {model_name}

# extra folder
EXTRA_DIR: str = "extra/"
MODULE_INFO_PATH: str = "extra/module_info.json"
