from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import sys


def is_available():
    return (hasattr(torch._C, "_c10d_init") and hasattr(torch._C, "_rpc_init")
            and hasattr(torch._C, "_dist_autograd_init"))


if is_available() and not (torch._C._c10d_init() and torch._C._rpc_init() and torch._C._dist_autograd_init()):
    raise RuntimeError("Failed to initialize PyTorch distributed support")


if is_available():
    from .distributed_c10d import *  # noqa: F401
    # Variables prefixed with underscore are not auto imported
    # See the comment in `distributed_c10d.py` above `_backend` on why we expose
    # this.
    from .distributed_c10d import _backend  # noqa: F401
    if sys.version_info >= (3, 0):
        from .rpc import _init_rpc
        from .rpc import *  # noqa: F401

        def init_model_parallel(self_name,
                                backend=RpcBackend.PROCESS_GROUP,
                                self_rank=-1,
                                init_method=None,
                                num_send_recv_threads=4):
            r"""
            Initializes model parallel primitives such as the local rpc agent
            and distributed autograd.

            Initializes the local RPC agent which immediately makes the current
            process ready to send and receive RPCs. The caller needs to make
            sure the specified backend is properly intialized before calling
            this method. For example, to use ``pg`` (ProcessGroup) backend,
            ``init_process_group`` must be invoked prior to this method.

            Arguments:
                backend (Enum): type of RPC backend implementation.
                            Currently, process group backend is the only
                            available backend implementation. (default:
                            ``RpcBackend.PROCESS_GROUP``).
                self_name (str): a globally unique name of this node. (e.g.,
                            ``Trainer3``, ``ParameterServer2``, ``Master``,
                            ``Worker1``) Name can only contain number, alphabet,
                            underscore, and/or dash, and must be shorter than
                            128 characters.
                self_rank (int): a globally unique id/rank of this node.
                init_method(str): backend specific init arguments.
                num_send_recv_threads(int): Number of threads for send/recv work.
            """
            _init_rpc(backend, self_name, self_rank, init_method, num_send_recv_threads)
            from .rpc import _agent
            autograd._init(_agent.get_worker_id().id)
