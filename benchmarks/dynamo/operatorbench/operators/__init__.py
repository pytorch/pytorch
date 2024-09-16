import os
import pathlib
import importlib
from typing import List
from utils.common import BenchmarkConfig

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

    def __init__(self, benchmark_config: BenchmarkConfig):
        self.benchmark_config = benchmark_config
        self.full_name = f"{self.name}.{self.variant}"

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
    names = map(lambda x: x.name, filter(lambda x: x.is_file(), dir.iterdir()))
    return file_name in names


def _list_operator_paths() -> List[str]:
    p = pathlib.Path(__file__).parent
    # Only load the model directories that contain a "__init.py__" file
    return sorted(
        str(child.absolute())
        for child in p.iterdir()
        if child.is_dir()
        and dir_contains_file(child, "__init__.py")
    )


def _load_valid_operators(module_path: str, operator_name: str) -> List:
    loaded_operators = []
    cls_name = "Operator"

    # Import the operator module
    try:
        operator_module = importlib.import_module(module_path, package=__name__)
        valid_operator_files = getattr(operator_module, "valid_operator_files", None)
        if valid_operator_files is None:
            raise ImportError(f"{module_path} does not define valid_operator_files")
    except ImportError as e:
        raise OperatorNotFoundError(f"Failed to load operator module {module_path}: {str(e)}")

    for file_name in valid_operator_files:
        tmp_file_name = file_name
        if file_name.endswith('.py'):
            tmp_file_name = file_name[:-3]
        operator_file_module_path = f"{module_path}.{tmp_file_name}"
        try:
            file_module = importlib.import_module(operator_file_module_path, package=__name__)
            Operator = getattr(file_module, cls_name, None)
            if Operator is None:
                print(f"Warning: {file_module} does not define attribute '{cls_name}', skipping.")
            else:
                if not hasattr(Operator, "name") or Operator.name is None:
                    Operator.name = f"{operator_name}"
                loaded_operators.append(Operator)
        except ImportError as e:
            print(f"Warning: Failed to load operator from {operator_file_module_path}: {str(e)}")
    return loaded_operators


def load_operator_by_name(operator_name: str) -> List[BaseOperator]:
    """
    Load all operator variants with the same operator name.

    This function searches for operators matching the provided name,
    imports their modules, and returns a list of loaded operator classes.

    Args:
        operator_name (str): The name of the operator to load.

    Raises:
        OperatorNotFoundError: If no operators are found with the given name.

    Returns:
        List[BaseOperator]: A list of loaded operator classes.
    """
    operators = filter(
        lambda x: operator_name.lower() == x.lower(),
        map(lambda y: os.path.basename(y), _list_operator_paths())
    )
    operators = list(operators)
    if not operators:
        raise OperatorNotFoundError(
            f"{operator_name} is not found in the operator list."
        )
    else:
        operator_name = operators[0]
        module_path = f".{operator_name}"
    assert (
        len(operators) == 1
    ), f"Found more than one operators {operators} with the exact name: {operator_name}"

    loaded_operators = _load_valid_operators(module_path, operator_name)
    return loaded_operators


def list_operators():
    operators = []
    for operator_path in _list_operator_paths():
        operator_name = os.path.basename(operator_path)
        module_path = f"operators.{operator_name}"
        loaded_operators = _load_valid_operators(module_path, operator_name)
        operators.extend(loaded_operators)
    return operators
