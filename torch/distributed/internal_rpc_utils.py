#!/usr/bin/env python3

import collections
import copyreg
import io
import pickle
import traceback
import threading
import sys

import torch

_thread_local_tensor_tables = threading.local()
_thread_local_tensor_tables.recv_tables = []
_thread_local_tensor_tables.send_tables = []


def _tensor_receiver(tensor_index):
    global _thread_local_tensor_tables
    return _thread_local_tensor_tables.recv_tables[tensor_index]

def _tensor_reducer(obj):
    global _thread_local_tensor_tables
    _thread_local_tensor_tables.send_tables.append(obj)
    tensor_index = len(_thread_local_tensor_tables.send_tables) - 1
    return (_tensor_receiver, (tensor_index, ))

if sys.version_info >= (3, 0):
    _dispatch_table = copyreg.dispatch_table.copy()
    _dispatch_table[torch.Tensor] = _tensor_reducer

def serialize(obj):
    f = io.BytesIO()
    p = pickle.Pickler(f)
    global _dispatch_table
    p.dispatch_table = _dispatch_table
    global _thread_local_tensor_tables
    _thread_local_tensor_tables.send_tables = []
    p.dump(obj)
    return f.getvalue()

# internal python function will be imported and executed in C++ land
# it unpickles pickled python udf strings and tensors and run the python
# udf, return serialized result and tensor tables
def run_python_udf_internal(pickled_python_udf, tensors):
    global _thread_local_tensor_tables
    _thread_local_tensor_tables.recv_tables = tensors
    python_udf = pickle.loads(pickled_python_udf)
    try:
        result = python_udf.func(*python_udf.args, **python_udf.kwargs)
    except Exception as e:
        # except str = exception info + traceback string
        except_str = "{}\n{}".format(repr(e), traceback.format_exc())
        result = RemoteException(except_str)
    return (serialize(result), _thread_local_tensor_tables.send_tables)

# internal python function will be imported and executed in C++ land
# it unpickled pickled python udf result and tensor tables, return python object
def load_python_udf_result_internal(pickled_python_result, tensors):
    global _thread_local_tensor_tables
    _thread_local_tensor_tables.recv_tables = tensors
    result = pickle.loads(pickled_python_result)
    if isinstance(result, RemoteException):
        raise Exception(result.msg)
    return result


PythonUDF = collections.namedtuple("PythonUDF", ["func", "args", "kwargs"])
RemoteException = collections.namedtuple("RemoteException", ["msg"])
