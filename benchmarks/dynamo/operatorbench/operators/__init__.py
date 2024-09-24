import importlib
import os
import pathlib
from typing import List
import types
import sys
import torch

from utils.common import BenchmarkConfig
from typing import Optional
from torch.utils._pytree import tree_map_only
from torch._inductor.utils import gen_gm_and_inputs
from torch._dynamo.backends.cudagraphs import cudagraphs_inner
from torch._inductor.compile_fx import compile_fx
from utils.metrics import Device

class OperatorNotFoundError(RuntimeError):
    pass


class BaseOperator:
    """
    Base class for operators.

    This class defines the structure for operator implementations.
    The forward, backward, full methods should **only contain**
    the code that users want to benchmark.
    """

    name = None
    variant = None
    benchmark_config = None
    full_name = None
    example_inputs_list = []
    
    def __init__(self, benchmark_config: BenchmarkConfig):
        self.benchmark_config = benchmark_config
        if self.full_name is None:
            self.full_name = f"{self.name}.{self.variant}"

    @classmethod
    def get_inputs(cls, benchmark_config: Optional[BenchmarkConfig] = None):
        if not cls.example_inputs_list:
            assert (
                benchmark_config is not None
            ), "Benchmark config is required to generate inputs"
            cls.generate_inputs(benchmark_config)
        return cls.example_inputs_list

    def forward(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def backward(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def full(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def single_run(self):
        """For the first input size"""
        raise NotImplementedError("Subclasses must implement this method.")


def dir_contains_file(dir, file_name) -> bool:
    # Use a generator expression instead of map
    names = (x.name for x in dir.iterdir() if x.is_file())
    return file_name in names


def _list_operator_paths() -> List[str]:
    p = pathlib.Path(__file__).parent
    # Only load the model directories that contain a "__init.py__" file
    return sorted(
        str(child.absolute())
        for child in p.iterdir()
        if child.is_dir() and dir_contains_file(child, "__init__.py")
    )


def _load_valid_operators(module_path: str, operator_name: str) -> List:
    loaded_operators = []
    cls_name = "Operator"

    # Import the operator module
    try:
        operator_module = importlib.import_module(module_path, package=__name__)
        # We only load the operator files that define the valid_operator_files attribute in the operator module
        valid_operator_files = getattr(operator_module, "valid_operator_files", None)
        if valid_operator_files is None:
            raise ImportError(f"{module_path} does not define valid_operator_files")
    except ImportError as e:
        raise OperatorNotFoundError(
            f"Failed to load operator module {module_path}: {str(e)}"
        ) from e

    for file_name in valid_operator_files:
        tmp_file_name = file_name
        if file_name.endswith(".py"):
            tmp_file_name = file_name[:-3]
        operator_file_module_path = f"{module_path}.{tmp_file_name}"
        try:
            file_module = importlib.import_module(
                operator_file_module_path, package=__name__
            )
            Operator = getattr(file_module, cls_name, None)
            if Operator is None:
                print(
                    f"Warning: {file_module} does not define attribute '{cls_name}', skipping."
                )
            else:
                if not hasattr(Operator, "name") or Operator.name is None:
                    Operator.name = f"{operator_name}"
                loaded_operators.append(Operator)
        except ImportError as e:
            print(
                f"Warning: Failed to load operator from {operator_file_module_path}: {str(e)}"
            )
    return loaded_operators


def list_operators():
    # This list is used to store all the operator classes, not instances
    operators = []
    for operator_path in _list_operator_paths():
        operator_name = os.path.basename(operator_path)
        module_path = f"operators.{operator_name}"
        loaded_operators = _load_valid_operators(module_path, operator_name)
        operators.extend(loaded_operators)
    return operators
