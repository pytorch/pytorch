#!/usr/bin/python3
import types
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import torch
import torch.distributed.rpc as rpc
from torch import Tensor, device, dtype, nn
from torch.distributed.nn.jit import instantiator
from torch.nn.parameter import Parameter
from torch.utils.hooks import RemovableHandle


_grad_t = Union[Tuple[Tensor, ...], Tensor]
# See https://mypy.readthedocs.io/en/latest/generics.html#generic-methods-and-generic-self for the use
# of `T` to annotate `self`. Many methods of `Module` return `self` and we want those return values to be
# the type of the subclass, not the looser type of `Module`.
T = TypeVar("T", bound="Module")

_NON_SCRIPTABLE_REMOTE_MODULE_MODULE = (
    instantiator.instantiate_non_scriptable_remote_module_template()
)


# RPC handler.
def _instantiate_template(module_interface_cls):
    instantiator.instantiate_scriptable_remote_module_template(module_interface_cls)


def _create_module(module_cls, args, kwargs, device="cpu", module_interface_cls=None):
    module = module_cls(*args, **kwargs)
    if not isinstance(module, nn.Module):
        raise ValueError(
            "Expect `module_cls(*args, **kwargs)` returns an instance of <class nn.Module>, "
            f"but it returns an instance of {type(module)}."
        )
    if module_interface_cls is not None:
        module = torch.jit.script(module)
    module.to(device)
    return rpc.RRef(module, module_interface_cls)


def _param_rrefs(module_rref, recurse):
    ret = []
    for param in module_rref.local_value().parameters(recurse):
        ret.append(rpc.RRef(param))
    return ret


def _raise_not_supported(name):
    raise ValueError("Method ``{}`` not supported for RemoteModule".format(name))


class _RemoteModule(nn.Module):
    def __init__(
        self,
        on: str,
        device: torch.device,
        module_cls: nn.Module,
        args: Tuple = None,
        kwargs: Dict[str, Any] = None,
        _module_interface_cls: Any = None,
    ):
        """
        A RemoteModule instance can only be created after RPC initialization.
        It creates a user-specified module on a specified remote node.
        It behaves like a regular ``nn.Module`` except that the ``forward`` method is
        executed on the remote node.
        It takes care of autograd recording to ensure the backward pass propogates
        gradients back to the corresponding remote module.

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

        Arguments:
            on (str or WorkerInfo): id or name of the destination worker.
            device (torch.device): Device on the destination worker where we‘d like to place this module.
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

            >>> # On worker 0:
            >>> import torch
            >>> import torch.distributed.rpc as rpc
            >>> from torch import nn, Tensor
            >>> from torch.distributed.nn.api.remote_module import RemoteModule
            >>>
            >>> rpc.init_rpc("worker0", rank=0, world_size=2)
            >>> remote_linear_module = RemoteModule(
            >>>     "worker1", "cpu", nn.Linear, args=(20, 30),
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

        # Sanity checks.
        assert rpc._is_current_rpc_agent_set(), "RemoteModule only works in RPC."

        # Default arguments preperation.
        args = args if args is not None else ()
        kwargs = kwargs if kwargs is not None else {}

        self.on = on

        if _module_interface_cls is not None:
            # Users reply on this field to know if this generated RemoteModule is TorchScript-able.
            self.is_scriptable = True

            # Instantiate template on remote side.
            fut = rpc.rpc_async(on, _instantiate_template, (_module_interface_cls,))

            # Instantiate template on local side.
            generated_module = instantiator.instantiate_scriptable_remote_module_template(
                _module_interface_cls
            )
            generated_methods = generated_module._generated_methods

            # Create the module on the remote side.
            fut.wait()  # Ensure remote_module_cls is available on remote side.
        else:
            self.is_scriptable = False
            generated_methods = _NON_SCRIPTABLE_REMOTE_MODULE_MODULE._generated_methods

        # Create the module on the remote side.
        self.module_rref = rpc.rpc_sync(
            on,
            _create_module,
            (module_cls, args, kwargs, device, _module_interface_cls),
        )

        # Install generated methods.
        for method in generated_methods:
            method_name = method.__name__
            method = torch.jit.export(method)
            setattr(self, method_name, types.MethodType(method, self))

    def remote_parameters(self, recurse: bool = True) -> List[rpc.RRef[Parameter]]:
        r"""Returns a list of RRefs of remote module parameters.
        This is typically passed to a distributed optimizer.
        Args:
            recurse (bool): if True, then returns parameters of the remote module
                and all submodules of the remote module.
                Otherwise, returns only parameters that are direct members of the remote module.

        Returns:
            A list of RRefs to remote module parameters.
        """
        return rpc.rpc_sync(self.on, _param_rrefs, args=(self.module_rref, recurse))

    def register_buffer(
        self, name: str, tensor: Optional[Tensor], persistent: bool = True
    ) -> None:
        _raise_not_supported(self.register_buffer.__name__)

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        _raise_not_supported(self.register_parameter.__name__)

    def add_module(self, name: str, module: Optional["Module"]) -> None:
        _raise_not_supported(self.add_module.__name__)

    def apply(self: T, fn: Callable[["Module"], None]) -> T:
        _raise_not_supported(self.apply.__name__)

    def cuda(self: T, device: Optional[Union[int, device]] = None) -> T:
        _raise_not_supported(self.cuda.__name__)

    def cpu(self: T) -> T:
        _raise_not_supported(self.cpu.__name__)

    def type(self: T, dst_type: Union[dtype, str]) -> T:
        _raise_not_supported(self.type.__name__)

    def float(self: T) -> T:
        _raise_not_supported(self.float.__name__)

    def double(self: T) -> T:
        _raise_not_supported(self.double.__name__)

    def half(self: T) -> T:
        _raise_not_supported(self.half.__name__)

    def bfloat16(self: T) -> T:
        _raise_not_supported(self.bfloat16.__name__)

    def to(self, *args, **kwargs):
        _raise_not_supported(self.to.__name__)

    def register_backward_hook(
        self, hook: Callable[["Module", _grad_t, _grad_t], Union[None, Tensor]]
    ) -> RemovableHandle:
        _raise_not_supported(self.register_backward_hook.__name__)

    def register_forward_pre_hook(self, hook: Callable[..., None]) -> RemovableHandle:
        _raise_not_supported(self.register_forward_pre_hook.__name__)

    def register_forward_hook(self, hook: Callable[..., None]) -> RemovableHandle:
        _raise_not_supported(self.register_forward_hook.__name__)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        _raise_not_supported(self.state_dict.__name__)

    def load_state_dict(
        self,
        state_dict: Union[Dict[str, Tensor], Dict[str, Tensor]],
        strict: bool = True,
    ):
        _raise_not_supported(self.load_state_dict.__name__)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        raise ValueError(
            "Method ``parameters`` not supported for RemoteModule. Please use ``remote_parameters`` instead."
        )

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, Tensor]]:
        _raise_not_supported(self.named_parameters.__name__)

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        _raise_not_supported(self.buffers.__name__)

    def named_buffers(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, Tensor]]:
        _raise_not_supported(self.named_buffers.__name__)

    def children(self) -> Iterator["Module"]:
        _raise_not_supported(self.children.__name__)

    def named_children(self) -> Iterator[Tuple[str, "Module"]]:
        _raise_not_supported(self.named_children.__name__)

    def modules(self) -> Iterator["Module"]:
        _raise_not_supported(self.modules.__name__)

    def named_modules(self, memo: Optional[Set["Module"]] = None, prefix: str = ""):
        _raise_not_supported(self.named_modules.__name__)

    def train(self: T, mode: bool = True) -> T:
        _raise_not_supported(self.train.__name__)

    def eval(self: T) -> T:
        _raise_not_supported(self.eval.__name__)

    def requires_grad_(self: T, requires_grad: bool = True) -> T:
        _raise_not_supported(self.requires_grad_.__name__)

    def zero_grad(self) -> None:
        _raise_not_supported(self.zero_grad.__name__)

    def share_memory(self: T) -> T:
        _raise_not_supported(self.share_memory.__name__)

    def extra_repr(self) -> str:
        _raise_not_supported(self.extra_repr.__name__)


class RemoteModule(_RemoteModule):
    """
        A RemoteModule instance can only be created after RPC initialization.
        It creates a user-specified module on a specified remote node.
        It behaves like a regular ``nn.Module`` except that the ``forward`` method is
        executed on the remote node.
        It takes care of autograd recording to ensure the backward pass propogates
        gradients back to the corresponding remote module.

        The arguments of ``forward_async`` and ``forward`` are the same as
        the ``forward`` method of the module returned by the ``module_cls``.

        For example, if ``module_cls`` returns an instance of ``nn.Linear``,
        that has ``forward`` method signature, ``def forward(input: Tensor) -> Tensor:``,
        the generated ``RemoteModule`` will have 2 methods in signature of
        ``def forward(input: Tensor) -> Tensor:`` and
        ``def forward_async(input: Tensor) -> Future[Tensor]:``.

    Arguments:
        to (str or WorkerInfo): id or name of the destination worker.
        device (torch.device): Device on the destination worker where we‘d like to place this module.
        module_cls (nn.Module): For example,
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

        >>> # On worker 0:
        >>> import torch
        >>> import torch.distributed.rpc as rpc
        >>> from torch import nn, Tensor
        >>> from torch.distributed.nn.api.remote_module import RemoteModule
        >>>
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> remote_linear_module = RemoteModule(
        >>>     "worker1", nn.Linear, args=(20, 30),
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

    def __init__(
        self,
        on: str,
        device: torch.device,
        module_cls: nn.Module,
        args: Tuple = None,
        kwargs: Dict[str, Any] = None,
    ):
        super().__init__(on, device, module_cls, args, kwargs)
