#!/usr/bin/env python3

import collections
import copyreg
import io
import pickle
import traceback

from . import set_current_rpc_dst


def serialize(obj, to=None):
    if to:
        set_current_rpc_dst(to)
    f = io.BytesIO()
    p = pickle.Pickler(f)
    p.dispatch_table = copyreg.dispatch_table.copy()
    p.dump(obj)
    return f.getvalue()


def run_python_udf_internal(pickled_python_udf, pickle_result=True):
    python_udf = pickle.loads(pickled_python_udf)
    try:
        result = python_udf.func(*python_udf.args, **python_udf.kwargs)
    except Exception as e:
        # except str = exception info + traceback string
        except_str = "{}\n{}".format(repr(e), traceback.format_exc())
        result = RemoteException(except_str)
    if pickle_result:
        return serialize(result)
    else:
        return result


def load_python_udf_result_internal(pickled_python_result):
    result = pickle.loads(pickled_python_result)
    if isinstance(result, RemoteException):
        raise Exception(result.msg)
    return result


PythonUDF = collections.namedtuple("PythonUDF", ["func", "args", "kwargs"])
RemoteException = collections.namedtuple("RemoteException", ["msg"])
