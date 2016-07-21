from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core

from caffe2.python.op.python_ops_python import \
    register, register_gradient


def CreatePythonOperator(f, inputs, outputs, grad_f=None, *args, **kwargs):
    token = register(f)
    if grad_f:
        register_gradient(token, grad_f)
    kwargs["token"] = token
    return core.CreateOperator("Python", inputs, outputs, *args, **kwargs)
