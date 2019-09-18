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

def tensor_reducer(obj):
    global _thread_local_tensor_tables
    _thread_local_tensor_tables.send_tables.append(obj)
    tensor_index = len(_thread_local_tensor_tables.send_tables) - 1
    return (_tensor_receiver, (tensor_index, ))

if sys.version_info >= (3, 0):
    _dispatch_table = copyreg.dispatch_table.copy()
    _dispatch_table[torch.Tensor] = tensor_reducer

def serialize(obj):
    f = io.BytesIO()
    p = pickle.Pickler(f)
    p.dispatch_table = _dispatch_table
    global _thread_local_tensor_tables
    _thread_local_tensor_tables.send_tables = []
    p.dump(obj)
    return f.getvalue()


def run_python_udf_internal(pickled_python_udf, tensors):
    _thread_local_tensor_tables.recv_tables = tensors
    python_udf = pickle.loads(pickled_python_udf)
    try:
        result = python_udf.func(*python_udf.args, **python_udf.kwargs)
    except Exception as e:
        # except str = exception info + traceback string
        except_str = "{}\n{}".format(repr(e), traceback.format_exc())
        result = RemoteException(except_str)
    return (serialize(result), _thread_local_tensor_tables.send_tables)


def load_python_udf_result_internal(pickled_python_result, tensors):
    _thread_local_tensor_tables.recv_tables = tensors
    result = pickle.loads(pickled_python_result)
    if isinstance(result, RemoteException):
        raise Exception(result.msg)
    return result


PythonUDF = collections.namedtuple("PythonUDF", ["func", "args", "kwargs"])
RemoteException = collections.namedtuple("RemoteException", ["msg"])
