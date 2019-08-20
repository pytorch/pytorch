#!/usr/bin/env python3

import pickle
import copyreg
import io
import collections

def serialize(obj):
    f = io.BytesIO()
    p = pickle.Pickler(f)
    p.dispatch_table = copyreg.dispatch_table.copy()
    p.dump(obj)
    return f.getvalue()

def run_python_udf_internal(pickled_python_udf):
    try:
        python_udf = pickle.loads(pickled_python_udf)
        result = python_udf.func(*python_udf.args, **python_udf.kwargs)
    except Exception as e:
        result = "run_python_udf_internal caught exception: " + str(e)
    return serialize(result)

def load_python_udf_result_internal(pickled_python_result):
    return pickle.loads(pickled_python_result)


PythonUDF = collections.namedtuple("PythonUDF", ["func", "args", "kwargs"])
