import torch
from torch import Tensor
import torch.utils._pytree as pytree
from collections import defaultdict
from torch.utils._python_dispatch import TorchDispatchMode

class OperatorProfiles:
    def __init__(self):
        self.data = defaultdict(list)

    def record(self, op, real_args, real_kwargs, real_output):
        metadata = pytree.tree_map_only(torch.Tensor, TensorMetadata, (real_args, real_kwargs, real_output))
        self.data[op].append(metadata)

    def generic_fake_kernel(self, op, fake_mode, *args, **kwargs):
        if op not in self.data:
            raise RuntimeError(f"no meta {op}")
        stuff =  self.data[op]

        def to_fake(metadata):
            fn = fake_mode.shape_env.create_unbacked_symint
            fake_shape = [fn() for _ in range(metadata.dim)]
            with fake_mode:
                return torch.empty(fake_shape, dtype=metadata.dtype, device=metadata.device)

        for s in stuff:
            if not profile_matches(s, *args, **kwargs):
                continue
            output_metadata = s[-1]
            result = pytree.tree_map_only(TensorMetadata, to_fake, output_metadata)
            return result
        raise RuntimeError(f"no meta {op}")


class TensorMetadata:
    def __init__(self, real_tensor):
        self.shape = real_tensor.shape
        self.strides = real_tensor.stride()
        self.storage_offset = real_tensor.storage_offset()
        self.is_contiguous = real_tensor.is_contiguous()
        self.dtype = real_tensor.dtype
        self.device = real_tensor.device
        # self.key_set = torch._C._key_set(real_tensor)

    @property
    def dim(self):
        return len(self.shape)

# torch_dispatch mode to record metas for this...

class OperatorProfilingMode(TorchDispatchMode):
    def __init__(self):
        self.profiles = OperatorProfiles()

    def reports(self):
        return self.profiles

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        out = func(*args, **kwargs)
        if not torch._library.utils.is_builtin(func):
            self.profiles.record(func, args, kwargs, out)
        # TODO: check that the op doesn't return a view.
        # If it does, we may need to error.
        # TODO: check that the op doesn't return any non-Tensors
        # (we don't support that)
        # TODO: block TorchBind (for now)
        return out


def profile_matches(profile, *fake_args, **fake_kwargs):
    args_metadata, kwargs_metadata, _ = profile
    # TODO: they're not zippable in general
    assert not kwargs_metadata
    assert not fake_kwargs
    for metadata, fake_arg in zip(args_metadata, fake_args):
        if isinstance(fake_arg, torch.Tensor):
            if fake_arg.device != metadata.device:
                return False
            if fake_arg.dtype != metadata.dtype:
                return False
            if fake_arg.dim() != metadata.dim:
                return False
            if not fake_arg.is_contiguous:
                return False
        else:
            if metadata != fake_arg:
                return False
    return True


# simple pointwise
@torch.library.custom_op("mylib::foo0", mutates_args={})
def foo0(x: Tensor) -> Tensor:
    return x.clone()

# output shape depends on the int passed in
@torch.library.custom_op("mylib::foo1", mutates_args={})
def foo1(x: Tensor, n: int) -> Tensor:
    return x.new_zeros(n)

# output shape depends on data
@torch.library.custom_op("mylib::foo2", mutates_args={})
def foo2(x: Tensor) -> Tensor:
    return x.nonzero()


def exercise_ops(x, y):
    result = []
    a = torch.ops.mylib.foo0(x)
    result.append(a)
    # b = torch.ops.mylib.foo1(x, 9)
    # result.append(b)
    # c = torch.ops.mylib.foo2(y)
    # result.append(c)
    return tuple(result)

x = torch.randn(2, 2)
y = torch.tensor([[0, 1, 2], [3, 4, 5]])
with OperatorProfilingMode() as mode:
    exercise_ops(x, y)
reports = mode.reports()

torch._dynamo.config.custom_ops_profile = reports

from functorch import make_fx
gm = make_fx(exercise_ops, tracing_mode="symbolic")(x, y)

torch.compile(exercise_ops)(x, y)
