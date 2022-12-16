import collections
import copyreg
import io
import pickle
import sys
import threading
import traceback
from enum import Enum

import torch
import torch.distributed as dist
from torch._C._distributed_rpc import _get_current_rpc_agent

__all__ = ["RPCExecMode", "serialize", "deserialize", "PythonUDF", "RemoteException"]

# Thread local tensor tables to store tensors while pickling torch.Tensor
# objects
_thread_local_tensor_tables = threading.local()
_pickler = pickle.Pickler
_unpickler = pickle.Unpickler


class RPCExecMode(Enum):
    SYNC = "sync"
    ASYNC = "async"
    ASYNC_JIT = "async_jit"
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
        # Ignore type error because dispatch_table is defined in third-party package
        self._dispatch_table = copyreg.dispatch_table.copy()  # type: ignore[attr-defined]
        self._dispatch_table[torch.Tensor] = self._tensor_reducer
        # Used for registering customized picklers.
        self._class_reducer_dict = {}

    def _register_reducer(self, obj_class, reducer):
        # For the same class, only register the reducer once.
        if obj_class not in self._class_reducer_dict:
            self._class_reducer_dict[obj_class] = reducer

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
    def _py_rref_receiver(cls, rref_fork_data):
        return dist.rpc.PyRRef._deserialize(rref_fork_data)

    def _py_rref_reducer(self, py_rref):
        rref_fork_data = py_rref._serialize()
        return (_InternalRPCPickler._py_rref_receiver, (rref_fork_data,))

    def _rref_reducer(self, rref):
        return self._py_rref_reducer(rref)

    @classmethod
    def _script_module_receiver(cls, script_module_serialized):
        """
        Given a serialized representation of a ScriptModule created with torch.jit.save,
        loads and returns the ScriptModule.
        """
        f = io.BytesIO(script_module_serialized)
        m = torch.jit.load(f)
        return m

    def _script_module_reducer(self, script_module):
        """
        Serializes a ScriptModule.
        """
        f = io.BytesIO()
        torch.jit.save(script_module, f)
        return (_InternalRPCPickler._script_module_receiver, (f.getvalue(),))

    def serialize(self, obj):
        r"""
        Serialize non tensor data into binary string, tensor data into
        tensor table
        """
        f = io.BytesIO()
        p = _pickler(f)
        p.dispatch_table = self._dispatch_table

        # rpc api could accept user picklers inheriting from _InternalRPCPickler to serialize rref,
        # user picklers could have different initialization function from _InternalRPCPickler,
        # but all the user picklers should call serialize() and use _rref_reducer to pickle rref
        # in python. also, when _internal_rpc_pickler is imported to rpc/api.py, rpc.RRef is not
        # compiled yet, it is not good place to acces rpc.RRef inside _InternalRPCPickler constructor,
        # so puting rref's dispatch table here
        #
        # The return value of a `rpc.remote(..)` call is type of `rpc.PyRRef`.
        # The deserialized RRef object on an RPC receiver side is type of `rpc.PyRRef`.
        # Ignore type error because dispatch_table is defined in third-party package
        p.dispatch_table[dist.rpc.PyRRef] = self._py_rref_reducer  # type: ignore[index]
        # An RRef created locally by RRef Python constructor is type of `rpc.RRef`.
        # Ignore type error because dispatch_table is defined in third-party package
        p.dispatch_table[dist.rpc.RRef] = self._rref_reducer  # type: ignore[index]

        # Add dispatch pickling for ScriptModule or its subclass.
        if isinstance(obj, torch.jit.ScriptModule):
            # Ignore type error because dispatch_table is defined in third-party package
            p.dispatch_table[obj.__class__] = self._script_module_reducer  # type: ignore[index]

        # Install customized picklers.
        for class_name in self._class_reducer_dict.keys():
            p.dispatch_table[class_name] = self._class_reducer_dict[class_name]  # type: ignore[index]

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
        Deserialize binary string + tensor table to original obj
        """
        # save _thread_local_tensor_tables.recv_tables if it is in nested call
        global _thread_local_tensor_tables
        if hasattr(_thread_local_tensor_tables, "recv_tables"):
            old_recv_tables = _thread_local_tensor_tables.recv_tables
        else:
            old_recv_tables = None
        _thread_local_tensor_tables.recv_tables = tensor_table

        try:
            unpickler = _unpickler(io.BytesIO(binary_data))
            ret = unpickler.load()
        except AttributeError as e:
            # Occurs when function is not found on module/class during
            # unpickling.
            except_str = (
                str(e)
                + """ Default RPC pickler does not serialize
            function code. Ensure that UDFs are defined on both caller and
            callee modules."""
            )
            ret = AttributeError(except_str)
            # Ensure the stack trace gets preserved
            ret.__cause__ = e

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
        except_str = (
            f"On {_get_current_rpc_agent().get_worker_info()}:\n"
            f"{repr(e)}\n{traceback.format_exc()}"
        )
        print(except_str, file=sys.stderr)
        result = RemoteException(except_str, type(e))
    return result


def _handle_exception(result):
    if isinstance(result, RemoteException):
        exception_msg = result.msg.encode("utf-8").decode("unicode_escape")
        # We wrap exception re-creation here in case some exception classes
        # cannot be constructed directly from a string.
        exc = None
        try:
            exc = result.exception_type(exception_msg)
        except BaseException as e:
            raise RuntimeError(  # noqa: B904
                f"Failed to create original exception type. Error msg was {str(e)}"
                f" Original exception on remote side was {exception_msg}"
            ) from e

        if exc is not None:
            raise exc


def _build_rpc_profiling_key(
    exec_type, func_name, current_worker_name, dst_worker_name
):
    """
    Builds the key that RPC calls are profiled with using the autograd profiler.
    This will be the name of the corresponding Event recorded in the profiler.

    Args:
        exec_type (RPCExecMode): Type of RPC/RRef call
        func_name (str): Name of function being profiled.
        current_worker_name (str): Name of current worker.
        dst_worker_name (str): Name of the destination worker.

    Returns:
        String representing profiling key
    """
    profile_key = "rpc_{rpc_type}#{func_name}({current_worker} -> {dst_worker})".format(
        rpc_type=exec_type.value,
        func_name=func_name,
        current_worker=current_worker_name,
        dst_worker=dst_worker_name,
    )
    return profile_key


def _start_record_function(exec_type, func_name, current_worker_name, dest_worker_name):
    """
    This function should be called from RPC/RRef functions to create a
    RecordFunction object for profiling. This function also runs the before
    callbacks that start the profiling, though the user is responsible for
    running the appropriate callbacks when the function to be profiled finishes.

    Args:
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
    rf = torch.autograd._RecordFunction()  # type: ignore[attr-defined]
    torch.autograd._run_before_callbacks(rf, profile_key)  # type: ignore[attr-defined]
    return rf


PythonUDF = collections.namedtuple("PythonUDF", ["func", "args", "kwargs"])
RemoteException = collections.namedtuple("RemoteException", ["msg", "exception_type"])
