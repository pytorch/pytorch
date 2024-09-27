import importlib
import os
import pathlib
import sys
import types
from typing import Dict, List, Optional

from utils.common import BenchmarkConfig
from utils.metrics import Device

import torch
from torch._dynamo.backends.cudagraphs import cudagraphs_inner
from torch._inductor.compile_fx import compile_fx
from torch._inductor.utils import gen_gm_and_inputs
from torch.utils._pytree import tree_map_only


class OperatorNotFoundError(RuntimeError):
    """Custom exception raised when an operator is not found."""


class BaseOperator:
    """
    Base class for operators.

    This class defines the structure for operator implementations.
    The forward, backward, full methods should **only contain**
    the code that users want to benchmark.

    Attributes:
        name (str): The main name of the operator, e.g. "FusedLinearCrossEntropy".
        variant (str): The variant of the operator, e.g. "baseline".
        benchmark_config (BenchmarkConfig): Configuration for the benchmark.
        full_name (str): The full name of the operator (name.variant). It is only valid for variants.
            It can be either assigned in the operator file or generated from name and variant.
        example_inputs_list (list): List of example inputs for the operator.
    """

    name = None
    variant = None
    benchmark_config = None
    full_name = None
    # example_inputs_list = []
    # example_input_used_config = None

    def __init__(self, benchmark_config: BenchmarkConfig):
        """
        Initialize the BaseOperator.

        Args:
            benchmark_config (BenchmarkConfig): Configuration for the benchmark.
        """
        self.benchmark_config = benchmark_config
        if self.full_name is None:
            self.full_name = f"{self.name}.{self.variant}"

    @classmethod
    def get_inputs(
        cls,
        input_mapping: Dict[str, List],
        benchmark_config: Optional[BenchmarkConfig] = None,
    ):
        """
        Get or generate example inputs for the operator.

        The format of the inputs is important and should meet the requirements
        of the operator. It is not necessary to have a unified format for
        different operators, but the format should be consistent within the
        same operator.

        This function is different from generate_inputs in that it does not
        generate inputs, but returns the inputs that have been generated in
        previous runs.

        Args:
            input_mapping (Dict[str, List]): Mapping from operator name to the input list.
            benchmark_config (Optional[BenchmarkConfig]): Configuration for the benchmark.

        Returns:
            list: List of example inputs.
        """
        if cls.name not in input_mapping:
            assert (
                benchmark_config is not None
            ), "Benchmark config is required to generate inputs"
            generated_inputs = cls.generate_inputs(benchmark_config)
            input_mapping[cls.name] = generated_inputs
        return input_mapping[cls.name]

    @classmethod
    def generate_inputs(cls, benchmark_config: BenchmarkConfig):
        """
        Generate example inputs for the operator. Each operator should implement
        this method and the format should be consistent with the operator.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def forward(self):
        """Perform the forward pass of the operator."""
        raise NotImplementedError("Subclasses must implement this method.")

    def backward(self):
        """Perform the backward pass of the operator. It can be bypassed if the operator does not have a backward pass."""
        raise NotImplementedError("Subclasses must implement this method.")

    def full(self):
        """Perform the full (forward + backward) pass of the operator."""
        raise NotImplementedError("Subclasses must implement this method.")

    def prepare_input_and_functions(self, input):
        """
        If needed, process the input before running the operator. This can be
        used to prepare the forward output for the backward benchmarking. By default,
        we return the input directly.

        Args:
            input: The input to the operator.

        Returns:
            The processed input.
        """
        return input


def _list_operator_paths() -> List[str]:
    """
    List the paths of all operator directories.

    Returns:
        List[str]: A sorted list of absolute paths to operator directories.
    """
    p = pathlib.Path(__file__).parent
    # Only load the model directories that contain a "__init.py__" file
    return sorted(
        str(child.absolute())
        for child in p.iterdir()
        if child.is_dir() and os.path.exists(os.path.join(child, "__init__.py"))
    )


def _load_valid_operators(module_path: str, operator_name: str) -> List:
    """
    Load valid operators from a given module path.

    Args:
        module_path (str): The path to the operator module.
        operator_name (str): The name of the operator.

    Returns:
        List: A list of loaded operator classes.

    Raises:
        OperatorNotFoundError: If the operator module fails to load.
    """
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
    """
    List all available operators. Each operator represents a variant of an base operator.

    Returns:
        List: A list of all operator classes.
    """
    # This list is used to store all the operator classes, not instances
    operators = []
    for operator_path in _list_operator_paths():
        operator_name = os.path.basename(operator_path)
        module_path = f"operators.{operator_name}"
        loaded_operators = _load_valid_operators(module_path, operator_name)
        operators.extend(loaded_operators)
    return operators
