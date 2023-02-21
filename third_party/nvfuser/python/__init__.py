# we need to import _C here to avoid confusing error message generated from failure in this python script ended up with complaining on `_C` not defined for `_C._FusionDefinition`
from . import _C
from ._C import *

class FusionDefinition(_C._FusionDefinition):
    def __enter__(self):
        return self._setup_definition()

    def __exit__(self, type, value, traceback):
        self._finalize_definition()

    def definition(self):
        raise NotImplementedError("definition() should be implemented by child class!")

    def schedule(self):
        raise NotImplementedError("schedule() should be implemented by child class!")

    def execute(self, inputs, **kwargs):
        """
        Executes an nvFuser set of kernels for a given Fusion

        Args:
            inputs (List[Union[Tensor, Scalar]]): A list of inputs to fusion.

        Kwargs:
            override_user_schedule (bool): For a user defined schedule, override with auto-generated schedule (default: False)

        Returns:
            List[Tensor]
        """
        override_user_schedule = kwargs.pop('override_user_schedule', False)
        func_based_def = False

        # if definition is not defined by a context manager, try a child class
        if self.id() is None:
            self._setup_definition()
            self.definition()
            self._finalize_definition()
            func_based_def = True

        # If schedule is defined by child class, make a schedule for inputs
        if func_based_def and (super(type(self), self).schedule != self.schedule):
            self._setup_schedule(inputs)
            self.schedule()
            self._finalize_schedule(inputs)

        return self._execute(inputs, override_user_schedule)

    def from_pytorch(self, tensor) :
        """
        Defines an nvfuser input tensor from a pytorch tensor
        
        Args:
            tensor (torch.Tensor): Input tensor to nvFuser

        Returns:
            nvfuser.Tensor
        """
        try:
            from .pytorch_utils import torch_dtype_to_nvfuser_dtype
        except ImportError:
            raise ImportError("Unable to import pytorch_utils!")

        if not tensor.is_cuda:
            raise ValueError("Tensor should be on a cuda device!")

        return self.define_tensor(sizes=tensor.size(), strides=tensor.stride(),
            dtype=torch_dtype_to_nvfuser_dtype(tensor.dtype))

from .nvfuser_version import __version__

def version():
    r"""returns nvfuser version in format of a string 'm.n.p+git[7d-sha]'.

    We strip the git[7d-sha] and convert the string to
    `nvfuser_version.Version` for comparison. e.g. you can use it as:
        import nvfuser
        print(nvfuser.version())              # 0.0.1+git21df524
        nvfuser.version() == '0.0.1`          # True
        nvfuser.version() > '0.0.0`           # True

        from nvfuser_version import Version
        nvfuser.version() < Version('1.0.0')  # True
    """
    return __version__
