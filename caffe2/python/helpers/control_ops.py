## @package control_ops
# Module caffe2.python.helpers.control_ops
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python.control_ops_util import add_if_op, add_while_op


def cond(model, cond_blob, external_blobs, then_model, else_model=None):
    """Condition"""
    add_if_op(
        model.net,
        cond_blob,
        external_blobs,
        then_model.net,
        else_model.net if else_model else None)


def loop(model, cond_blob, external_blobs, loop_model, cond_model=None):
    """Loop"""
    add_while_op(
        model.net,
        cond_blob,
        external_blobs,
        loop_model.net,
        cond_model.net if cond_model else None)
