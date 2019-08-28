#!/usr/bin/env python3

import collections
import copyreg
import io
import pickle
import traceback


def serialize(obj):
    f = io.BytesIO()
    p = pickle.Pickler(f)
    p.dispatch_table = copyreg.dispatch_table.copy()
    p.dump(obj)
    return f.getvalue()


def run_python_udf_internal(pickled_python_udf):
    python_udf = pickle.loads(pickled_python_udf)
    try:
        result = python_udf.func(*python_udf.args, **python_udf.kwargs)
    except Exception as e:
        # except str = exception info + traceback string
        except_str = "{}\n{}".format(repr(e), traceback.format_exc())
        result = RemoteException(except_str)
    return serialize(result)


def load_python_udf_result_internal(pickled_python_result):
    result = pickle.loads(pickled_python_result)
    if isinstance(result, RemoteException):
        raise Exception(result.msg)
    return result


PythonUDF = collections.namedtuple("PythonUDF", ["func", "args", "kwargs"])
RemoteException = collections.namedtuple("RemoteException", ["msg"])
