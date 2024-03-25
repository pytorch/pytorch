#!/usr/bin/python3
import collections
import io
import sys
import types
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import torch
import torch.distributed.rpc as rpc
from torch import Tensor, device, dtype, nn
from torch.distributed.nn.jit import instantiator
from torch.distributed import _remote_device
from torch.distributed.rpc.internal import _internal_rpc_pickler
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.utils.hooks import RemovableHandle

__all__ = ["RemoteModule"]

_grad_t = Union[Tuple[Tensor, ...], Tensor]
# See https://mypy.readthedocs.io/en/latest/generics.html#generic-methods-and-generic-self for the use
# of `T` to annotate `self`. Many methods of `Module` return `self` and we want those return values to be
# the type of the subclass, not the looser type of `Module`.
T = TypeVar("T", bound="Module")

_NON_SCRIPTABLE_REMOTE_MODULE_MODULE = (
    instantiator.instantiate_non_scriptable_remote_module_template()
)

_REMOTE_MODULE_PICKLED_ATTRIBUTES = (
    "on",
    "device",
    "is_device_map_set",
    "is_scriptable",
    "generated_methods",
    "module_rref",
)

_SerializedRemoteModule = collections.namedtuple("_SerializedRemoteModule", _REMOTE_MODULE_PICKLED_ATTRIBUTES)  # type: ignore[misc]

# These attributes are mostly from RemoteModule's parent class and are intentionally not pickled.
# A new attribute of RemoteModule should be either in _REMOTE_MODULE_PICKLED_ATTRIBUTES
# or _REMOTE_MODULE_ATTRIBUTES_IGNORE_FOR_PICKLING.
# Otherwise, it will not be pickled.
_REMOTE_MODULE_ATTRIBUTES_IGNORE_FOR_PICKLING = (
    "training",
    "_parameters",
    "_buffers",
    "_non_persistent_buffers_set",
    "_backward_hooks",
    "_backward_pre_hooks",
    "_is_full_backward_hook",
    "_forward_hooks",
    "_forward_hooks_with_kwargs",
    "_forward_hooks_always_called",
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs",
    "_state_dict_hooks",
    "_state_dict_pre_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
    "_state_dict_pre_hooks",
    "_modules",
    # The two attributes below are generated methods, not available at pickling time.
    "forward_async",
    "forward",
)


# RPC handler.
def _instantiate_template(module_interface_cls, enable_moving_cpu_tensors_to_cuda):
    instantiator.instantiate_scriptable_remote_module_template(
        module_interface_cls, enable_moving_cpu_tensors_to_cuda
    )


def _create_module(module_cls, args, kwargs, device):
    module = module_cls(*args, **kwargs)
    if not isinstance(module, nn.Module):
        raise ValueError(
            "Expect `module_cls(*args, **kwargs)` returns an instance of <class nn.Module>, "
            f"but it returns an instance of {type(module)}."
        )
    module.to(device)
    return module


def _create_module_with_interface(
    module_cls, args, kwargs, device, module_interface_cls
):
    module = _create_module(module_cls, args, kwargs, device)
    if module_interface_cls is not None:
        module = torch.jit.script(module)
    return rpc.RRef(module, module_interface_cls)


def _param_rrefs(module_rref, recurse) -> List[rpc.RRef[Parameter]]:
    ret: List[rpc.RRef[Parameter]] = []
    for param in module_rref.local_value().parameters(recurse):
        ret.append(rpc.RRef(param))
    return ret


def _raise_not_supported(name: str) -> None:
    raise ValueError(f"Method ``{name}`` not supported for RemoteModule")


class _RemoteModule(nn.Module):

    def __new__(cls, *args, **kwargs):
        # Use __new__ for logging purposes.
        torch._C._log_api_usage_once("torch.distributed.nn.api.remote_module")
        return super().__new__(cls)

    def __init__(
        self,
        remote_device: str,
        module_cls: Type[nn.Module],
        args: Optional[Tuple] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        _module_interface_cls: Any = None,
    ):
        """
        RemoteModule instance can only be created after RPC initialization.

        It creates a user-specified module on a specified remote node.
        It behaves like a regular ``nn.Module`` except that the ``forward`` method is
        executed on the remote node.
        It takes care of autograd recording to ensure the backward pass propagates
        gradients back to the corresponding remote module.
        It can be shared across processors using `RPC framework <https://pytorch.org/docs/stable/rpc.html>`__,
        without incurring any overheads of copying the actual module,
        which is equivalent to an :class:`~torch.distributed.rpc.RRef`
        pointing to the remote module.

        The arguments of ``forward_async`` and ``forward`` are the same as
        the ``forward`` method of the module returned by the ``module_cls``.

        Apart from ``forward_async`` and ``forward``, no other methods are supported from nn.Module for now.

        Particularly, to create a hybrid model, typically the local modules should be
        created outside of remote modules, rather than as submodules of any remote module (by calling ``add_module``).
        Hybrid Example:
                >>> class HybridModel(nn.Module):
                >>>     def __init__(self):
                >>>         nn.Module.__init__(self)
                >>>         self.remote_embedding = RemoteModule(...)
                >>>         self.local_linear = nn.Linear(...)

        For example, if ``module_cls`` returns an instance of ``nn.Linear``,
        that has ``forward`` method signature, ``def forward(input: Tensor) -> Tensor:``,
        the generated ``RemoteModule`` will have 2 methods in signature of
        ``def forward(input: Tensor) -> Tensor:`` and
        ``def forward_async(input: Tensor) -> Future[Tensor]:``.

        .. note::
            If the remote module is placed on a cuda device,
            any input CPU tensors will be automatically moved to the same cuda device,
            and GPU tensors are returned over the wire according to the device map of the remote worker on TensorPipe RPC backend.

        Args:
            remote_device (str): Device on the destination worker where we'd like to place this module.
                The device can be a local device or a remote device specified by one of the following remote
                formats:

                    1. "rank:<rank>/<device>" (ex: "rank:0/cuda:0").
                    2. "<worker_name>/<device>" (ex: "trainer0/cuda:0").

                In addition, the device field can be optional and the default value is "cpu".
            module_cls (nn.Module): For example,
                >>> class MyModule(nn.Module):
                >>>     def forward(input):
                >>>         return input + 1
                >>>
                >>> module_cls = MyModule
            args (Sequence, optional): args to be passed to ``module_cls``.
            kwargs (Dict, optional): kwargs to be passed to ``module_cls``.
            _module_interface_cls (type, optional): The TorchScript interface type for the module
                to be created. The type object should be decorated by @torch.jit.interface.
                If not provided, the generated RemoteModule is not torchscript-able.
                Warning, this is an experimental API and susceptible to frequent changes.

        Returns:
            A remote module instance which wraps the :class:`~nn.Module` created by the
            user-provided ``module_cls``, it has a blocking ``forward`` method and an
            asynchronous ``forward_async`` method that returns a future of the ``forward`` call
            on the user-provided module on the remote side.

        Example::
            Run the following code in two different processes:

            >>> # xdoctest: +SKIP("distributed")
            >>> # On worker 0:
            >>> import torch
            >>> import torch.distributed.rpc as rpc
            >>> from torch import nn, Tensor
            >>> from torch.distributed.nn.api.remote_module import RemoteModule
            >>>
            >>> rpc.init_rpc("worker0", rank=0, world_size=2)
            >>> remote_linear_module = RemoteModule(
            >>>     "worker1/cpu", nn.Linear, args=(20, 30),
            >>> )
            >>> input = torch.randn(128, 20)
            >>> ret_fut = remote_linear_module.forward_async(input)
            >>> ret = ret_fut.wait()
            >>> rpc.shutdown()

            >>> # On worker 1:
            >>> import torch
            >>> import torch.distributed.rpc as rpc
            >>>
            >>> rpc.init_rpc("worker1", rank=1, world_size=2)
            >>> rpc.shutdown()
        """
        super().__init__()

        enable_moving_cpu_tensors_to_cuda = self._prepare_init(remote_device)

        # Default arguments preparation.
        args = args if args is not None else ()
        kwargs = kwargs if kwargs is not None else {}

        if _module_interface_cls is not None:
            # Users reply on this field to know if this generated RemoteModule is TorchScript-able.
            self.is_scriptable = True

            # Instantiate template on remote side.
            fut = rpc.rpc_async(
                self.on,
                _instantiate_template,
                (_module_interface_cls, enable_moving_cpu_tensors_to_cuda),
            )

            self._init_template(
                _module_interface_cls, enable_moving_cpu_tensors_to_cuda
            )

            # Instantiate template on remote side.
            fut = rpc.rpc_async(
                self.on,
                _instantiate_template,
                (_module_interface_cls, enable_moving_cpu_tensors_to_cuda),
            )

            # Create the module on the remote side.
            fut.wait()  # Ensure remote_module_cls is available on remote side.

            # TODO: We need to change this to rpc.remote, and make it async (see the else branch below).
            # For that we need to be able to apply _module_interface_cls to the RRef returned by rpc.remote
            # See https://github.com/pytorch/pytorch/issues/58098 for more context.
            self.module_rref = rpc.rpc_sync(
                self.on,
                _create_module_with_interface,
                (module_cls, args, kwargs, self.device, _module_interface_cls),
            )
        else:
            self.is_scriptable = False
            self.generated_methods = (
                _NON_SCRIPTABLE_REMOTE_MODULE_MODULE._generated_methods
            )
            # Create the module on the remote side.
            self.module_rref = rpc.remote(
                self.on,
                _create_module,
                (module_cls, args, kwargs, self.device),
            )

        self._install_generated_methods()
        self._check_attribute_picklability()

    def remote_parameters(self, recurse: bool = True) -> List[rpc.RRef[Parameter]]:
        """
        Return a list of :class:`~torch.distributed.rpc.RRef` pointing to the remote module's parameters.

        This can typically be used in conjunction
        with :class:`~torch.distributed.optim.DistributedOptimizer`.

        Args:
            recurse (bool): if True, then returns parameters of the remote
                module and all submodules of the remote module. Otherwise,
                returns only parameters that are direct members of the
                remote module.

        Returns:
            A list of :class:`~torch.distributed.rpc.RRef` (``List[RRef[nn.Parameter]]``)
            to remote module's parameters.
        """
        return rpc.rpc_sync(self.on, _param_rrefs, args=(self.module_rref, recurse))

    def get_module_rref(self) -> rpc.RRef[nn.Module]:
        """Return an :class:`~torch.distributed.rpc.RRef` (``RRef[nn.Module]``) pointing to the remote module."""
        return self.module_rref

    @torch.jit.export
    def __getstate__(self):
        raise RuntimeError(
            "Cannot pickle RemoteModule in python pickler. RemoteModule can only be pickled when using RPC"
        )

    @torch.jit.export
    def __setstate__(self, state):
        raise RuntimeError(
            "Cannot unpickle RemoteModule in python pickler. RemoteModule can only be unpickled when using RPC"
        )

    def register_buffer(
        self, name: str, tensor: Optional[Tensor], persistent: bool = True
    ) -> None:
        _raise_not_supported(self.register_buffer.__name__)

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        _raise_not_supported(self.register_parameter.__name__)

    def add_module(self, name: str, module: Optional[Module]) -> None:
        _raise_not_supported(self.add_module.__name__)

    def apply(self: T, fn: Callable[[Module], None]) -> T:  # type: ignore[return]
        _raise_not_supported(self.apply.__name__)

    def cuda(self: T, device: Optional[Union[int, device]] = None) -> T:  # type: ignore[return]
        _raise_not_supported(self.cuda.__name__)

    def ipu(self: T, device: Optional[Union[int, device]] = None) -> T:  # type: ignore[return]
        _raise_not_supported(self.ipu.__name__)

    def xpu(self: T, device: Optional[Union[int, device]] = None) -> T:  # type: ignore[return]
        _raise_not_supported(self.xpu.__name__)

    def cpu(self: T) -> T:  # type: ignore[return]
        _raise_not_supported(self.cpu.__name__)

    def type(self: T, dst_type: Union[dtype, str]) -> T:  # type: ignore[return]
        _raise_not_supported(self.type.__name__)

    def float(self: T) -> T:  # type: ignore[return]
        _raise_not_supported(self.float.__name__)

    def double(self: T) -> T:  # type: ignore[return]
        _raise_not_supported(self.double.__name__)

    def half(self: T) -> T:  # type: ignore[return]
        _raise_not_supported(self.half.__name__)

    def bfloat16(self: T) -> T:  # type: ignore[return]
        _raise_not_supported(self.bfloat16.__name__)

    def to(self, *args, **kwargs) -> T:  # type: ignore[misc, return, type-var]
        _raise_not_supported(self.to.__name__)

    def register_backward_hook(  # type: ignore[return]
        self, hook: Callable[[Module, _grad_t, _grad_t], Union[None, _grad_t]]
    ) -> RemovableHandle:
        _raise_not_supported(self.register_backward_hook.__name__)

    def register_forward_pre_hook(  # type: ignore[return]
        self,
        hook: Union[
            Callable[[T, Tuple[Any, ...]], Optional[Any]],
            Callable[[T, Tuple[Any, ...], Dict[str, Any]], Optional[Tuple[Any, Dict[str, Any]]]],
        ],
        prepend: bool = False,
        with_kwargs: bool = False,
    ) -> RemovableHandle:
        _raise_not_supported(self.register_forward_pre_hook.__name__)

    def register_forward_hook(  # type: ignore[return, override]
        self,
        hook: Union[
            Callable[[T, Tuple[Any, ...], Any], Optional[Any]],
            Callable[[T, Tuple[Any, ...], Dict[str, Any], Any], Optional[Any]],
        ],
        prepend: bool = False,
        with_kwargs: bool = False,
    ) -> RemovableHandle:
        _raise_not_supported(self.register_forward_hook.__name__)

    def state_dict(self, *args, **kwargs):
        _raise_not_supported(self.state_dict.__name__)

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        _raise_not_supported(self.load_state_dict.__name__)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        raise ValueError(
            "Method ``parameters`` not supported for RemoteModule. Please use ``remote_parameters`` instead."
        )

    def named_parameters(  # type: ignore[return]
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
        _raise_not_supported(self.named_parameters.__name__)

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:  # type: ignore[return]
        _raise_not_supported(self.buffers.__name__)

    def named_buffers(  # type: ignore[return]
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Tensor]]:
        _raise_not_supported(self.named_buffers.__name__)

    def children(self) -> Iterator[Module]:  # type: ignore[return]
        _raise_not_supported(self.children.__name__)

    def named_children(self) -> Iterator[Tuple[str, Module]]:  # type: ignore[return]
        _raise_not_supported(self.named_children.__name__)

    def modules(self) -> Iterator[Module]:  # type: ignore[return]
        _raise_not_supported(self.modules.__name__)

    def named_modules(
        self,
        memo: Optional[Set[Module]] = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ):
        _raise_not_supported(self.named_modules.__name__)

    def train(self: T, mode: bool = True) -> T:
        return self.module_rref.rpc_sync().train()  # type: ignore[operator, union-attr]

    def eval(self: T) -> T:
        return self.module_rref.rpc_sync().eval()  # type: ignore[operator, union-attr]

    def requires_grad_(self: T, requires_grad: bool = True) -> T:  # type: ignore[return]
        _raise_not_supported(self.requires_grad_.__name__)

    def zero_grad(self, set_to_none: bool = True) -> None:
        _raise_not_supported(self.zero_grad.__name__)

    def share_memory(self: T) -> T:  # type: ignore[return]
        _raise_not_supported(self.share_memory.__name__)

    def extra_repr(self) -> str:  # type: ignore[return]
        _raise_not_supported(self.extra_repr.__name__)

    def _prepare_init(self, remote_device_str: str) -> bool:
        """Prepare the initialization and returns whether to enable automatically moving CPU tensors to CUDA devices."""
        # Sanity check.
        assert rpc._is_current_rpc_agent_set(), "RemoteModule only works in RPC."

        remote_device = _remote_device(remote_device_str)
        self.on = remote_device.worker_name() if remote_device.worker_name() is not None else remote_device.rank()
        self.device = str(remote_device.device())
        agent = rpc._get_current_rpc_agent()
        # If the device map of the remote worker is set,
        # then enable moving any input CPU tensors to the same cuda device.
        self.is_device_map_set = bool(
            agent._get_device_map(agent.get_worker_info(self.on))  # type: ignore[arg-type]
        )
        # ``enable_moving_cpu_tensors_to_cuda`` is less strict than ``is_device_map_set``:
        # If ``enable_moving_cpu_tensors_to_cuda`` is true, but the device map is not set,
        # then any CPU tensors can still be moved to a cuda device to run forward,
        # but the output must be moved back to CPU before being sent over the wire.
        enable_moving_cpu_tensors_to_cuda = torch.device(self.device).type == "cuda"
        return enable_moving_cpu_tensors_to_cuda

    def _init_template(self, module_interface_cls, enable_moving_cpu_tensors_to_cuda):
        """Instantiate template on local side."""
        generated_module = instantiator.instantiate_scriptable_remote_module_template(
            module_interface_cls, enable_moving_cpu_tensors_to_cuda
        )
        self.generated_methods = generated_module._generated_methods

    def _check_attribute_picklability(self):
        """Check if all the attribute has explicitly defined whether to be pickled (i.e., picklability)."""
        for k in self.__dict__.keys():
            if (
                k not in _REMOTE_MODULE_PICKLED_ATTRIBUTES
                and k not in _REMOTE_MODULE_ATTRIBUTES_IGNORE_FOR_PICKLING
            ):
                raise AttributeError(
                    f"Attribute {k} must be either in ``_REMOTE_MODULE_PICKLED_ATTRIBUTES`` or "
                    "``_REMOTE_MODULE_ATTRIBUTES_IGNORE_FOR_PICKLING``."
                )

    def _install_generated_methods(self):
        for method in self.generated_methods:
            method_name = method.__name__
            method = torch.jit.export(method)
            setattr(self, method_name, types.MethodType(method, self))

    @staticmethod
    def init_from_module_rref(
        remote_device: str,
        module_rref: rpc.RRef[nn.Module],
        _module_interface_cls: Any = None,
    ):
        """
        Besides the constructor, a RemoteModule instance can also be initialized given a module RRef.

        This alternate initialization method can be particularly useful if we want to create multiple
        RemoteModule instances that share the same underlying module and reduce memory consumption.

        Moreover, this also provides a workaround for passing script RemoteModule over RPC,
        which is not supported. The recommended way is as follows:

            1. the sender creates a RemoteModule;
            2. the sender sends its ``module_rref`` over RPC;
            3. the receiver calls this method to initialize another RemoteModule using the same ``module_rref``.

        Example::
            Run the following code in two different processes:

            >>> # xdoctest: +SKIP("distributed")
            >>> # On worker 0:
            >>> import torch
            >>> import torch.distributed.rpc as rpc
            >>> from torch import nn, Tensor
            >>> from torch.distributed.nn.api.remote_module import RemoteModule
            >>>
            >>> rpc.init_rpc("worker0", rank=0, world_size=2)
            >>> remote_module = RemoteModule(
            >>>     "worker1/cpu", nn.Linear, args=(20, 30),
            >>> )
            >>>
            >>> remote_module1 = rpc.rpc_sync(
            >>>     "worker1/cpu",
            >>>     RemoteModule.init_from_module_rref,
            >>>     ("worker1/cpu", remote_module1.get_module_rref()),
            >>> )
            >>> rpc.shutdown()

            >>> # On worker 1:
            >>> import torch
            >>> import torch.distributed.rpc as rpc
            >>>
            >>> rpc.init_rpc("worker1", rank=1, world_size=2)
            >>> rpc.shutdown()

        Args:
            remote_device (str): Device on the destination worker where we'd like to place this module.
                The device can be a local device or a remote device specified by one of the following remote
                formats:

                    1. "rank:<rank>/<device>" (ex: "rank:0/cuda:0").
                    2. "<worker_name>/<device>" (ex: "trainer0/cuda:0").

                In addition, the device field can be optional and the default value is "cpu".
            module_rref (RRef[nn.Module]): The module reference shared by both the caller and
                the created remote module.
            _module_interface_cls (type, optional): The TorchScript interface type for the module
                to be created. The type object should be decorated by @torch.jit.interface.
                If not provided, the generated RemoteModule is not torchscript-able.
                Warning, this is an experimental API and susceptible to frequent changes.

        Returns:
            A remote module instance which wraps the :class:`~nn.Module` created by the
            user-provided ``module_rref``, it has a blocking ``forward`` method and an
            asynchronous ``forward_async`` method that returns a future of the ``forward`` call
            on the user-provided module on the remote side.
        """
        # NOTE: if a new attribute is added to this class, also need to add it
        # to ``_REMOTE_MODULE_PICKLED_ATTRIBUTES`` for pickling/unpickling.

        remote_module = object.__new__(RemoteModule)

        enable_moving_cpu_tensors_to_cuda = remote_module._prepare_init(remote_device)

        if _module_interface_cls is not None:
            # Users reply on this field to know if this generated RemoteModule is TorchScript-able.
            remote_module.is_scriptable = True

            remote_module._init_template(
                _module_interface_cls, enable_moving_cpu_tensors_to_cuda
            )
        else:
            remote_module.is_scriptable = False
            remote_module.generated_methods = (
                _NON_SCRIPTABLE_REMOTE_MODULE_MODULE._generated_methods
            )
        remote_module.module_rref = module_rref

        remote_module._install_generated_methods()
        remote_module._check_attribute_picklability()

        return remote_module


class RemoteModule(_RemoteModule):
    """
        A RemoteModule instance can only be created after RPC initialization.

        It creates a user-specified module on a specified remote node.
        It behaves like a regular ``nn.Module`` except that the ``forward`` method is
        executed on the remote node.
        It takes care of autograd recording to ensure the backward pass propagates
        gradients back to the corresponding remote module.

        It generates two methods ``forward_async`` and ``forward`` based on the
        signature of the ``forward`` method of ``module_cls``. ``forward_async``
        runs asynchronously and returns a Future. The arguments of ``forward_async``
        and ``forward`` are the same as the ``forward`` method of the module
        returned by the ``module_cls``.

        For example, if ``module_cls`` returns an instance of ``nn.Linear``,
        that has ``forward`` method signature: ``def forward(input: Tensor) -> Tensor:``,
        the generated ``RemoteModule`` will have 2 methods with the signatures:

        | ``def forward(input: Tensor) -> Tensor:``
        | ``def forward_async(input: Tensor) -> Future[Tensor]:``

    Args:
        remote_device (str): Device on the destination worker where we'd like to place this module.
            The format should be "<workername>/<device>", where the device field can be parsed as torch.device type.
            E.g., "trainer0/cpu", "trainer0", "ps0/cuda:0".
            In addition, the device field can be optional and the default value is "cpu".
        module_cls (nn.Module): Class for the module to be created remotely. For example,

            >>> class MyModule(nn.Module):
            >>>     def forward(input):
            >>>         return input + 1
            >>>
            >>> module_cls = MyModule

        args (Sequence, optional): args to be passed to ``module_cls``.
        kwargs (Dict, optional): kwargs to be passed to ``module_cls``.

    Returns:
        A remote module instance which wraps the :class:`~nn.Module` created by the
        user-provided ``module_cls``, it has a blocking ``forward`` method and an
        asynchronous ``forward_async`` method that returns a future of the ``forward`` call
        on the user-provided module on the remote side.

    Example::
        Run the following code in two different processes:

        >>> # xdoctest: +SKIP("distributed")
        >>> # On worker 0:
        >>> import torch
        >>> import torch.distributed.rpc as rpc
        >>> from torch import nn, Tensor
        >>> from torch.distributed.nn.api.remote_module import RemoteModule
        >>>
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> remote_linear_module = RemoteModule(
        >>>     "worker1/cpu", nn.Linear, args=(20, 30),
        >>> )
        >>> input = torch.randn(128, 20)
        >>> ret_fut = remote_linear_module.forward_async(input)
        >>> ret = ret_fut.wait()
        >>> rpc.shutdown()

        >>> # On worker 1:
        >>> import torch
        >>> import torch.distributed.rpc as rpc
        >>>
        >>> rpc.init_rpc("worker1", rank=1, world_size=2)
        >>> rpc.shutdown()

        Furthermore, a more practical example that is combined with
        `DistributedDataParallel <https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel>`__ (DDP)
        can be found in this `tutorial <https://pytorch.org/tutorials/advanced/rpc_ddp_tutorial.html>`__.
    """

    def __init__(
        self,
        remote_device: str,
        module_cls: Type[nn.Module],
        args: Optional[Tuple] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(remote_device, module_cls, args, kwargs)


def _remote_module_receiver(
    *remote_module_pickled_attrs,
):
    """Deserializes a RemoteModule."""
    serialized_remote_module = _SerializedRemoteModule._make(
        remote_module_pickled_attrs
    )
    m = object.__new__(RemoteModule)
    m.__dict__.update(serialized_remote_module._asdict())

    # Unpickling the attribute `module_rref` must invoke RRef's `_deserialize()` method.
    m.module_rref = rpc.PyRRef._deserialize(m.module_rref)

    # Install generated methods when unpickled.
    for method in m.generated_methods:
        method_name = method.__name__
        method = torch.jit.export(method)
        setattr(m, method_name, types.MethodType(method, m))

    return m


def _remote_module_reducer(remote_module):
    """Serialize a RemoteModule."""
    pickled_attrs = {}
    for k, v in remote_module.__dict__.items():
        # Pickling the attribute `module_rref` must invoke RRef's `_serialize()` method.
        if k == "module_rref":
            pickled_attrs[k] = v._serialize()
        elif k in _REMOTE_MODULE_PICKLED_ATTRIBUTES:
            pickled_attrs[k] = v
        # Check if unpickled attributes are all in _REMOTE_MODULE_ATTRIBUTES_IGNORE_FOR_PICKLING.
        elif k not in _REMOTE_MODULE_ATTRIBUTES_IGNORE_FOR_PICKLING:
            print(
                f"The new attribute ``{k}`` of RemoteModule is ignored during RPC pickling. "
                "To pickle this attribute, please add it to ``_REMOTE_MODULE_PICKLED_ATTRIBUTES``. "
                "Otherwise, please explicitly add it to ``_REMOTE_MODULE_ATTRIBUTES_IGNORE_FOR_PICKLING``.",
                file=sys.stderr,
            )

    return (
        _remote_module_receiver,
        tuple(pickled_attrs.values()),
    )


def _recursive_script_module_receiver(
    recursive_script_module_serialized,
):
    """Deserializes a RecursiveScriptModule that does not contain a script RemoteModule."""
    f = io.BytesIO(recursive_script_module_serialized)
    m = torch.jit.load(f)
    return m


def _recursive_script_module_reducer(recursive_script_module):
    """Serialize a RecursiveScriptModule that does not contain a script RemoteModule, and raises an error otherwise."""
    if hasattr(recursive_script_module._c, "module_rref"):
        raise RuntimeError(
            "Passing a script RemoteModule over RPC is not supported. Please create a RemoteModule in the sender, "
            "send the `module_rref` to the receiver, and create a new instance on the receiver end by passing this `module_rref`."
        )

    f = io.BytesIO()
    torch.jit.save(recursive_script_module, f)
    return (_recursive_script_module_receiver, (f.getvalue(),))


_internal_rpc_pickler._register_reducer(RemoteModule, _remote_module_reducer)
_internal_rpc_pickler._register_reducer(
    torch.jit.RecursiveScriptModule, _recursive_script_module_reducer
)
