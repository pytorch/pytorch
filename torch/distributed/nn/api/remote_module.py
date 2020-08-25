#!/usr/bin/python3
import types
from typing import Any, Dict, Tuple

import torch
import torch.distributed.rpc as rpc
from torch import nn
from torch.distributed.nn.jit import instantiator


_NON_SCRIPTABLE_REMOTE_MODULE_MODULE = (
    instantiator.instantiate_non_scriptable_remote_module_template()
)


# RPC handler.
def _instantiate_template(module_interface_cls):
    instantiator.instantiate_scriptable_remote_module_template(module_interface_cls)


def _create_module(module_cls, args, kwargs, module_interface_cls=None):
    module = module_cls(*args, **kwargs)
    if not isinstance(module, nn.Module):
        raise ValueError(
            "Expect `module_cls(*args, **kwargs)` returns an instance of <class nn.Module>, "
            f"but it returns an instance of {type(module)}."
        )
    if module_interface_cls is not None:
        module = torch.jit.script(module)
    return rpc.RRef(module, module_interface_cls)


class _RemoteModule(nn.Module):
    def __init__(
        self,
        on: str,
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

        For example, if ``module_cls`` returns an instance of ``nn.Linear``,
        that has ``forward`` method signature, ``def forward(input: Tensor) -> Tensor:``,
        the generated ``RemoteModule`` will have 2 methods in signature of
        ``def forward(input: Tensor) -> Tensor:`` and
        ``def forward_async(input: Tensor) -> Future[Tensor]:``.

        Arguments:
            on (str or WorkerInfo): id or name of the destination worker.
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
        super().__init__()

        # Sanity checks.
        assert rpc._is_current_rpc_agent_set(), "RemoteModule only works in RPC."

        # Default arguments preperation.
        args = args if args is not None else ()
        kwargs = kwargs if kwargs is not None else {}

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
            (module_cls, args, kwargs, _module_interface_cls),
        )

        # Install generated methods.
        for method in generated_methods:
            method_name = method.__name__
            method = torch.jit.export(method)
            setattr(self, method_name, types.MethodType(method, self))


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
        module_cls: nn.Module,
        args: Tuple = None,
        kwargs: Dict[str, Any] = None,
    ):
        super().__init__(on, module_cls, args, kwargs)
