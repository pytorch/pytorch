import importlib
import os
import pathlib
from typing import List
import types
import sys
import torch

from utils.common import BenchmarkConfig
from .operator_inp_utils import OperatorInputsLoader, to_channels_last
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

    def prepare_input_and_functions(self, input):
        return input

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


def list_operators(benchmark_config: BenchmarkConfig):
    # This list is used to store all the operator classes, not instances
    operators = []
    for operator_path in _list_operator_paths():
        operator_name = os.path.basename(operator_path)
        module_path = f"operators.{operator_name}"
        loaded_operators = _load_valid_operators(module_path, operator_name)
        operators.extend(loaded_operators)
    operators.extend(dynamically_create_native_operator_classes(benchmark_config))
    return operators


def dynamically_create_native_operator_classes(benchmark_config: BenchmarkConfig):
    """
    To keep same with custom operators, we dynamically create operator classes here.
    """
    timm_loader = OperatorInputsLoader.get_timm_loader()
    huggingface_loader = OperatorInputsLoader.get_huggingface_loader()
    torchbench_loader = OperatorInputsLoader.get_torchbench_loader()
    all_ops = list(timm_loader.get_all_ops()) + list(huggingface_loader.get_all_ops()) + \
        list(torchbench_loader.get_all_ops())
    # remove duplicate operators
    all_ops = list(set(all_ops))

    def merge_inputs(cls, benchmark_config: BenchmarkConfig):
        """
        We don't differentiate inputs for different suite any more.
        """
        op_eval = cls.op_eval
        inps_gens = []
        if str(op_eval) in timm_loader.operator_db:
            inps_gens.append(timm_loader.get_inputs_for_operator(
                op_eval, dtype=benchmark_config.dtype, device=benchmark_config.device.value))
        if str(op_eval) in huggingface_loader.operator_db:
            inps_gens.append(huggingface_loader.get_inputs_for_operator(
                op_eval, dtype=benchmark_config.dtype, device=benchmark_config.device.value))
        if str(op_eval) in torchbench_loader.operator_db:
            inps_gens.append(torchbench_loader.get_inputs_for_operator(
                op_eval, dtype=benchmark_config.dtype, device=benchmark_config.device.value))
        input_list = []
        num_samples = min(benchmark_config.max_samples, 1000000)
        index = 0
        while index < num_samples:
            for inp_gen in inps_gens:
                try:
                    inps = next(inp_gen)
                    input_list.append((inps, None))
                    index += 1
                except StopIteration:
                    break
        cls.example_inputs_list = input_list

    def prepare_input_and_functions(self, input):
        input0 = input[0]
        args, kwargs = input0
        if self.benchmark_config.channels_last:
            args, kwargs = tree_map_only(
                torch.Tensor, to_channels_last, (args, kwargs)
            )

        gm, gm_args = gen_gm_and_inputs(self.op_eval, args, kwargs)
        torch.jit._builtins._register_builtin(
            torch.ops.aten.convolution_backward.default, "aten::convolution_backward"
        )
        if self.benchmark_config.device == Device.CUDA:
            if self.variant == "Eager":
                cudagraphs_eager = cudagraphs_inner(
                    gm, gm_args, copy_outputs=False, copy_inputs=False
                )
                self.forward = cudagraphs_eager
                self.full = cudagraphs_eager
            elif self.variant == "Inductor":
                compiled_fn = compile_fx(gm, gm_args)
                cudagraphs_compiled = cudagraphs_inner(
                    compiled_fn, gm_args, copy_outputs=False, copy_inputs=False
                )
                self.forward = cudagraphs_compiled
                self.full = cudagraphs_compiled
        else:
            if self.variant == "Eager":
                self.forward = gm
                self.full = gm
            elif self.variant == "Inductor":
                compiled_fn = compile_fx(gm, gm_args)
                self.forward = compiled_fn
                self.full = compiled_fn
        return gm_args

    operators = []
    for op_eval in all_ops:
        class_name = f"native_{str(op_eval).replace('.', '_')}"
        # create a new module for each operator
        op_name_module = types.ModuleType(f'operators.{class_name}')
        sys.modules[f'operators.{class_name}'] = op_name_module
        # create a new module for each varient to help with code organization and printing
        eager_module = types.ModuleType(f'operators.{class_name}.Eager')
        sys.modules[f'operators.{class_name}.Eager'] = eager_module
        inductor_module = types.ModuleType(f'operators.{class_name}.Inductor')
        sys.modules[f'operators.{class_name}.Inductor'] = inductor_module
        # the new class for operator, and it is the parent class for all its variants
        new_op_class = type(class_name, (BaseOperator,), {})
        # need the loaders to generate inputs for the same operator
        new_op_class.huggingface_loader = huggingface_loader
        new_op_class.torchbench_loader = torchbench_loader
        new_op_class.timm_loader = timm_loader
        new_op_class.op_eval = op_eval
        new_op_class.name = str(op_eval)
        new_op_class.generate_inputs = classmethod(merge_inputs)
        # create eager and inductor variants classes
        new_eager_op_class = type(f"{class_name}.Eager.Operator", (new_op_class,), {})
        new_eager_op_class.variant = "Eager"
        new_eager_op_class.full_name = f"{new_eager_op_class.name}.Eager"
        new_eager_op_class.prepare_input_and_functions = prepare_input_and_functions
        eager_module.Operator = new_eager_op_class
        new_inductor_op_class = type(f"{class_name}.Inductor.Operator", (new_op_class,), {})
        new_inductor_op_class.variant = "Inductor"
        new_inductor_op_class.full_name = f"{new_inductor_op_class.name}.Inductor"
        new_inductor_op_class.prepare_input_and_functions = prepare_input_and_functions
        inductor_module.Operator = new_inductor_op_class
        operators.append(new_eager_op_class)
        operators.append(new_inductor_op_class)
    return operators
