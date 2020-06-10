#!/usr/bin/python3
import types
import uuid
from typing import Any, Callable, Dict, Tuple

import torch
import torch.distributed.rpc as rpc
from torch import nn
from torch.distributed.nn.jit import instantiator


_NON_SCRIPTABLE_REMOTE_MODULE_MODULE = (
    instantiator.instantiate_non_scriptable_remote_module_template()
)


def _gen_global_unique_name():
    return f"{uuid.uuid4().hex}"


# RPC handler.
def _instantiate_template(module_interface_cls):
    instantiator.instantiate_scriptable_remote_module_template(module_interface_cls)


def _module_creator_wrapper(
    module_creator, args, kwargs, module_interface_cls=None
):
    module = module_creator(*args, **kwargs)
    if not isinstance(module, nn.Module):
        raise ValueError(
            "Expect module_creator returns an instancee of <class nn.Module>, "
            f"but it returns an instance of {type(module)}."
        )
    if module_interface_cls is not None:
        module = torch.jit.script(module)
    return rpc.RRef(module, module_interface_cls)


class _RemoteModule(nn.Module):
    def __init__(
        self,
        on: str,
        module_creator: Callable,
        args: Tuple = None,
        kwargs: Dict[str, Any] = None,
        global_unique_name: str = None,
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
        the ``forward`` method of the module returned by the ``module_creator``.

        For example, if ``module_creator`` returns an instace of ``nn.Linear``,
        that has ``forward`` method signature, ``def forward(input: Tensor) -> Tensor:``,
        the generated ``RemoteModule`` will have 2 methods in signature of
        ``def forward(input: Tensor) -> Tensor:`` and
        ``def forward_async(input: Tensor) -> Future[Tensor]:``.

        Arguments:
            on (str or WorkerInfo): id or name of the destination worker.
            module_creator (Callable): A ``module_creator`` could be
                1. A type object that is subclass of ``nn.Module``.
                    For example,
                    >>> class MyModule(nn.Module):
                    >>>     def forward(input):
                    >>>         return input + 1
                    >>>
                    >>> module_creator = MyModule
                2. A function that returns a instance of ``nn.Module``.
                    For example,
                    >>> def module_creator():
                    >>>     module = MyModule()
                    >>>     scripted_module = torch.jit.script(module)
                    >>>     return scripted_module
            args (Sequence, optional): args to be passed to ``module_creator``.
            kwargs (Dict, optional): kwargs to be passed to ``module_creator``.
            global_unique_name (str, optional): The unique name of the created RemoteModule,
                useful for profiling purpose. If not provided, a UUID4 will
                be generated as its name.
            _module_interface_cls (type, optional): The TorchScript interface type for the module
                to be created. The type object should be decorated by @torch.jit.interface.
                If not provided, the generated RemoteModule is not torchscript-able.
                Warning, this is an experimental API and susceptible to frequent changes.

        Returns:
            A remote module instance which wraps the :class:`~nn.Module` created by the
            user-provided ``module_creator``, it has a blocking ``forward`` method and an
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

        # Users can specify a unique name for the generated RemoteModule.
        # For example, a RemoteModule represents a shard of a EmbeddingBag.
        # In the case of re-sharding after resuming from a checkpoint.
        # The shard can be referenced using this unique name.
        self.global_unique_name = (  # Assign a global name for the module to be created.
            global_unique_name
            if global_unique_name is not None
            else _gen_global_unique_name()
        )

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
            _module_creator_wrapper,
            (module_creator, args, kwargs, _module_interface_cls),
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
        the ``forward`` method of the module returned by the ``module_creator``.

        For example, if ``module_creator`` returns an instace of ``nn.Linear``,
        that has ``forward`` method signature, ``def forward(input: Tensor) -> Tensor:``,
        the generated ``RemoteModule`` will have 2 methods in signature of
        ``def forward(input: Tensor) -> Tensor:`` and
        ``def forward_async(input: Tensor) -> Future[Tensor]:``.

    Arguments:
        to (str or WorkerInfo): id or name of the destination worker.
        module_creator (Callable): A ``module_creator`` could be
            1. A type object that is subclass of ``nn.Module``.
                For example,
                >>> class MyModule(nn.Module):
                >>>     def forward(input):
                >>>         return input + 1
                >>>
                >>> module_creator = MyModule
            2. A function that returns a instance of ``nn.Module``.
                For example,
                >>> def module_creator():
                >>>     module = MyModule()
                >>>     scripted_module = torch.jit.script(module)
                >>>     return scripted_module
        args (Sequence, optional): args to be passed to ``module_creator``.
        kwargs (Dict, optional): kwargs to be passed to ``module_creator``.
        global_unique_name (str, optional): The unique name of the created RemoteModule,
            useful for profiling purpose. If not provided, a UUID4 will
            be generated as its name.

    Returns:
        A remote module instance which wraps the :class:`~nn.Module` created by the
        user-provided ``module_creator``, it has a blocking ``forward`` method and an
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
        module_creator: Callable,
        args: Tuple = None,
        kwargs: Dict[str, Any] = None,
        global_unique_name: str = None,
    ):
        super().__init__(
            on, module_creator, args, kwargs, global_unique_name=global_unique_name
        )
