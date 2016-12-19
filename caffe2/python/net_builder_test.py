from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import workspace
from caffe2.python.core import Plan, to_execution_step
from caffe2.python.net_builder import ops, NetBuilder
import unittest


def test_loop():
    x = ops.Const(5)
    y = ops.Const(0)
    with ops.loop():
        ops.stop_if(ops.EQ([x, ops.Const(0)]))
        ops.Add([x, ops.Const(-1)], [x])
        ops.Add([y, ops.Const(1)], [y])
    return y


def test_inner_stop(x):
    ops.stop_if(ops.LT([x, ops.Const(5)]))


def test_outer():
    x = ops.Const(10)
    # test stop_if(False)
    with ops.stop_guard() as g1:
        test_inner_stop(x)

    # test stop_if(True)
    y = ops.Const(3)
    with ops.stop_guard() as g2:
        test_inner_stop(y)

    # test no stop
    with ops.stop_guard() as g4:
        ops.Const(0)

    # test empty clause
    with ops.stop_guard() as g3:
        pass

    return (
        g1.has_stopped(), g2.has_stopped(), g3.has_stopped(), g4.has_stopped())


def test_if(x):
    y = ops.Const(1)
    with ops.If(ops.GT([x, ops.Const(50)])):
        ops.Const(2, blob_out=y)
    with ops.If(ops.LT([x, ops.Const(50)])):
        ops.Const(3, blob_out=y)
        ops.stop()
        ops.Const(4, blob_out=y)
    return y


class TestNetBuilder(unittest.TestCase):
    def test_ops(self):
        with NetBuilder() as nb:
            y = test_loop()
            z, w, a, b = test_outer()
            p = test_if(ops.Const(75))
            q = test_if(ops.Const(25))
        plan = Plan('name')
        plan.AddStep(to_execution_step(nb))
        ws = workspace.C.Workspace()
        ws.run(plan)
        expected = [
            (y, 5),
            (z, False),
            (w, True),
            (a, False),
            (b, False),
            (p, 3),
            (q, 2),
        ]
        for b, expected in expected:
            actual = ws.blobs[str(b)].fetch()
            self.assertEquals(actual, expected)
