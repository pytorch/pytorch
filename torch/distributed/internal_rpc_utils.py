#!/usr/bin/env python3

import collections
import copyreg
import io
import pickle
import traceback
import threading
import six

import torch

# Thread local tensor tables to store tensors while pickling torch.Tensor
# objects
_thread_local_tensor_tables = threading.local()

# This class provides serialize() and deserialize() interfaces to serialize
# data to be "binary string + tensor table" format
# So for RPC python UDF function and args, non tensor data will be serialized
# into regular binary string, tensor data will be put into thread local tensor
# tables, this serialization format is consistent with builtin operator and args
# using JIT pickler. This format will make tensor handling in C++ much easier,
# e.g. attach tensor to distributed autograd graph in C++
class InternalRPCPickler:
    def __init__(self):
        # python2 does not have dispatch_table, add "if six.PY3" condition,
        # as InternalRPCPickler still got build in python2 even
        # we skipped python 2 tests for rpc_test
        if six.PY3:
            self._dispatch_table = copyreg.dispatch_table.copy()
            self._dispatch_table[torch.Tensor] = self._tensor_reducer

    @classmethod
    def _tensor_receiver(cls, tensor_index):
        global _thread_local_tensor_tables
        return _thread_local_tensor_tables.recv_tables[tensor_index]

    def _tensor_reducer(self, obj):
        global _thread_local_tensor_tables
        _thread_local_tensor_tables.send_tables.append(obj)
        tensor_index = len(_thread_local_tensor_tables.send_tables) - 1
        return (InternalRPCPickler._tensor_receiver, (tensor_index, ))

    # Serialize non tensor data into binary string, tensor data into
    # tensor table
    def serialize(self, obj):
        f = io.BytesIO()
        p = pickle.Pickler(f)
        p.dispatch_table = self._dispatch_table

        # save _thread_local_tensor_tables.send_tables if it is in nested call
        global _thread_local_tensor_tables
        if hasattr(_thread_local_tensor_tables, "send_tables"):
            should_restore = True
            old_send_tables = _thread_local_tensor_tables.send_tables
        else:
            should_restore = False
        _thread_local_tensor_tables.send_tables = []

        p.dump(obj)

        # restore _thread_local_tensor_tables.send_tables if return
        # from nested call, otherwise clean up the table
        tensors = _thread_local_tensor_tables.send_tables
        if should_restore:
            _thread_local_tensor_tables.send_tables = old_send_tables
        else:
            del _thread_local_tensor_tables.send_tables

        return (f.getvalue(), tensors)

    # Deserilize binary string + tensor table to original obj
    def deserialize(self, binary_data, tensor_table):
        # save _thread_local_tensor_tables.recv_tables if it is in nested call
        global _thread_local_tensor_tables
        if hasattr(_thread_local_tensor_tables, "recv_tables"):
            should_restore = True
            old_recv_tables = _thread_local_tensor_tables.recv_tables
        else:
            should_restore = False
        _thread_local_tensor_tables.recv_tables = tensor_table

        ret = pickle.loads(binary_data)

        # restore _thread_local_tensor_tables.recv_tables if return
        # from nested call, otherwise clean up the table
        if should_restore:
            _thread_local_tensor_tables.recv_tables = old_recv_tables
        else:
            del _thread_local_tensor_tables.recv_tables

        return ret

# Create _internal_rpc_pickler only once to initialize _dispatch_table only once
_internal_rpc_pickler = InternalRPCPickler()


# Internal python function will be imported and executed in C++ land
# it unpickles pickled python udf strings and tensors and run the python
# udf, return serialized result and tensor tables
def run_python_udf_internal(pickled_python_udf, tensors):
    python_udf = _internal_rpc_pickler.deserialize(pickled_python_udf, tensors)
    try:
        result = python_udf.func(*python_udf.args, **python_udf.kwargs)
    except Exception as e:
        # except str = exception info + traceback string
        except_str = "{}\n{}".format(repr(e), traceback.format_exc())
        result = RemoteException(except_str)
    return _internal_rpc_pickler.serialize(result)


# Internal python function will be imported and executed in C++ land
# it unpickled pickled python udf result and tensor tables, return python object
def load_python_udf_result_internal(pickled_python_result, tensors):
    result = _internal_rpc_pickler.deserialize(pickled_python_result, tensors)
    if isinstance(result, RemoteException):
        raise Exception(result.msg)
    return result


PythonUDF = collections.namedtuple("PythonUDF", ["func", "args", "kwargs"])
RemoteException = collections.namedtuple("RemoteException", ["msg"])
