import collections
import copyreg
from enum import Enum
import io
import pickle
import threading
import traceback

import torch
import torch.distributed as dist


# Thread local tensor tables to store tensors while pickling torch.Tensor
# objects
_thread_local_tensor_tables = threading.local()


class RPCExecMode(Enum):
    SYNC = "sync"
    ASYNC = "async"
    REMOTE = "remote"

class _InternalRPCPickler:
    r"""
    This class provides serialize() and deserialize() interfaces to serialize
    data to be "binary string + tensor table" format
    So for RPC python UDF function and args, non tensor data will be serialized
    into regular binary string, tensor data will be put into thread local tensor
    tables, this serialization format is consistent with builtin operator and args
    using JIT pickler. This format will make tensor handling in C++ much easier,
    e.g. attach tensor to distributed autograd graph in C++
    """

    def __init__(self):
        # python2 does not have dispatch_table, add "if torch._six.PY3" condition,
        # as _InternalRPCPickler still got build in python2 even
        # we skipped python 2 tests for rpc_test
        if torch._six.PY3:
            self._dispatch_table = copyreg.dispatch_table.copy()
            self._dispatch_table[torch.Tensor] = self._tensor_reducer

    @classmethod
    def _tensor_receiver(cls, tensor_index):
        global _thread_local_tensor_tables
        return _thread_local_tensor_tables.recv_tables[tensor_index]

    def _tensor_reducer(self, tensor):
        global _thread_local_tensor_tables
        _thread_local_tensor_tables.send_tables.append(tensor)
        tensor_index = len(_thread_local_tensor_tables.send_tables) - 1
        return (_InternalRPCPickler._tensor_receiver, (tensor_index,))

    @classmethod
    def _rref_receiver(cls, rref_fork_data):
        return dist.rpc.RRef._deserialize(rref_fork_data)

    def _rref_reducer(self, rref):
        rref_fork_data = rref._serialize()
        return (_InternalRPCPickler._rref_receiver, (rref_fork_data, ))

    def serialize(self, obj):
        r"""
        Serialize non tensor data into binary string, tensor data into
        tensor table
        """
        f = io.BytesIO()
        p = pickle.Pickler(f)
        p.dispatch_table = self._dispatch_table

        # rpc api could accept user picklers inheriting from _InternalRPCPickler to serialize rref,
        # user picklers could have different initialization function from _InternalRPCPickler,
        # but all the user picklers should call serialize() and use _rref_reducer to pickle rref
        # in python. also, when _internal_rpc_pickler is imported to rpc/api.py, rpc.RRef is not
        # compiled yet, it is not good place to acces rpc.RRef inside _InternalRPCPickler constructor,
        # so puting rref's dispatch table here
        p.dispatch_table[dist.rpc.RRef] = self._rref_reducer

        # save _thread_local_tensor_tables.send_tables if it is in nested call
        global _thread_local_tensor_tables
        if hasattr(_thread_local_tensor_tables, "send_tables"):
            old_send_tables = _thread_local_tensor_tables.send_tables
        else:
            old_send_tables = None
        _thread_local_tensor_tables.send_tables = []

        p.dump(obj)

        # restore _thread_local_tensor_tables.send_tables if return
        # from nested call, otherwise clean up the table
        tensors = _thread_local_tensor_tables.send_tables
        if old_send_tables is not None:
            _thread_local_tensor_tables.send_tables = old_send_tables
        else:
            del _thread_local_tensor_tables.send_tables

        return (f.getvalue(), tensors)

    def deserialize(self, binary_data, tensor_table):
        r"""
        Deserilize binary string + tensor table to original obj
        """
        # save _thread_local_tensor_tables.recv_tables if it is in nested call
        global _thread_local_tensor_tables
        if hasattr(_thread_local_tensor_tables, "recv_tables"):
            old_recv_tables = _thread_local_tensor_tables.recv_tables
        else:
            old_recv_tables = None
        _thread_local_tensor_tables.recv_tables = tensor_table

        try:
            ret = pickle.loads(binary_data)
        except AttributeError as e:
            # Occurs when function is not found on module/class during
            # unpickling.
            except_str = str(e) + """ Default RPC pickler does not serialize
            function code. Ensure that UDFs are defined on both caller and
            callee modules."""
            ret = AttributeError(except_str)

        # restore _thread_local_tensor_tables.recv_tables if return
        # from nested call, otherwise clean up the table
        if old_recv_tables is not None:
            _thread_local_tensor_tables.recv_tables = old_recv_tables
        else:
            del _thread_local_tensor_tables.recv_tables

        return ret


# Create _internal_rpc_pickler only once to initialize _dispatch_table only once
_internal_rpc_pickler = _InternalRPCPickler()


def serialize(obj):
    return _internal_rpc_pickler.serialize(obj)


def deserialize(binary_data, tensor_table):
    return _internal_rpc_pickler.deserialize(binary_data, tensor_table)


def _run_function(python_udf):
    r"""
    This function is exclusively called from C++.
    See ``torch/csrc/distributed/rpc/python_rpc_handler.cpp``.

    Runs a Python UDF and returns its return value.
    Wraps any exception in ``RemoteException`` if the function raises.
    """
    try:
        if isinstance(python_udf, AttributeError):
            raise python_udf
        result = python_udf.func(*python_udf.args, **python_udf.kwargs)
    except Exception as e:
        # except str = exception info + traceback string
        except_str = "{}\n{}".format(repr(e), traceback.format_exc())
        result = RemoteException(except_str, type(e))
    return result


def _handle_exception(result):
    if isinstance(result, RemoteException):
        raise result.exception_type(result.msg)


def _start_record_function(exec_type, func_name, current_worker_name, dest_worker_name):
    """
    This function should be called from RPC/RRef functions to create a
    RecordFunction object for profiling. This function also runs the before
    callbacks that start the profiling, though the user is responsible for
    running the appropriate callbacks when the function to be profiled finishes.

    Arguments:
        exec_type (RPCExecMode): Type of RPC/RRef call
        func_name (str): Name of function being profiled.
        current_worker_name (str): Name of current worker.
        dest_worker_name (str): Name of the destination worker.

    Returns:
        An instance of `torch.autograd._RecordFunction`.
    """
    assert torch.autograd._profiler_enabled(), "Autograd profiler should be enabled."
    profile_key = "rpc_{}#{}({} -> {})".format(
        exec_type.value, str(func_name), current_worker_name, dest_worker_name
    )
    rf = torch.autograd._RecordFunction()
    torch.autograd._run_before_callbacks(rf, profile_key)
    return rf


PythonUDF = collections.namedtuple("PythonUDF", ["func", "args", "kwargs"])
RemoteException = collections.namedtuple("RemoteException", ["msg", "exception_type"])
