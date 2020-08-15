from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core, workspace
from caffe2.python.test_util import TestCase

import unittest


class TestAtomicOps(TestCase):
    @unittest.skip("Test is flaky: https://github.com/pytorch/pytorch/issues/28179")
    def test_atomic_ops(self):
        """
        Test that both countdown and checksum are update atomically by having
        cowntdown count from 20k to 0 from parallel the workers and updating
        the checksum to the value fetched. If operations are trully atomic,
        each value from 1 to 20k should be fetched exactly once from the
        countdown, and fed exactly once to the checksum, such that at the end
        checksum must contain the exact value of sum[i=0..20000](i).
        """
        init_net = core.Net('init')
        mutex_countdown = init_net.CreateMutex([])
        mutex_checksum = init_net.CreateMutex([])
        countdown = init_net.ConstantFill([], shape=[], value=20000,
                                          dtype=core.DataType.INT32)
        checksum = init_net.ConstantFill(
            [], shape=[], value=0, dtype=core.DataType.INT32)
        minus_one = init_net.ConstantFill(
            [], shape=[], value=-1, dtype=core.DataType.INT32)
        steps = []
        for i in range(0, 100):
            net = core.Net('net:%d' % i)
            _, fetched_count = net.AtomicFetchAdd(
                [mutex_countdown, countdown, minus_one],
                [countdown, 'fetched_count:%d' % i])
            net.AtomicFetchAdd(
                [mutex_checksum, checksum, fetched_count],
                [checksum, 'not_used'])
            steps.append(
                core.execution_step('worker:%d' % i, net, num_iter=200))
        super_step = core.execution_step(
            'parent', steps, concurrent_substeps=True)
        plan = core.Plan('plan')
        plan.AddStep(core.execution_step('init', init_net))
        plan.AddStep(super_step)
        workspace.RunPlan(plan)
        # checksum = sum[i=1..20000](i) = 20000 * 20001 / 2 = 200010000
        self.assertEquals(workspace.FetchBlob(checksum), 200010000)

    @unittest.skip("Test is flaky: https://github.com/pytorch/pytorch/issues/28179")
    def test_atomic64_ops(self):
        """
        Test that both countdown and checksum are update atomically by having
        cowntdown count from 20k to 0 from parallel the workers and updating
        the checksum to the value fetched. If operations are trully atomic,
        each value from 1 to 20k should be fetched exactly once from the
        countdown, and fed exactly once to the checksum, such that at the end
        checksum must contain the exact value of sum[i=0..20000](i).
        """
        init_net = core.Net('init')
        mutex_countdown = init_net.CreateMutex([])
        mutex_checksum = init_net.CreateMutex([])
        countdown = init_net.ConstantFill([], shape=[], value=20000,
                                          dtype=core.DataType.INT64)
        checksum = init_net.ConstantFill(
            [], shape=[], value=0, dtype=core.DataType.INT64)
        minus_one = init_net.ConstantFill(
            [], shape=[], value=-1, dtype=core.DataType.INT64)
        steps = []
        for i in range(0, 100):
            net = core.Net('net:%d' % i)
            _, fetched_count = net.AtomicFetchAdd64(
                [mutex_countdown, countdown, minus_one],
                [countdown, 'fetched_count:%d' % i])
            net.AtomicFetchAdd64(
                [mutex_checksum, checksum, fetched_count],
                [checksum, 'not_used'])
            steps.append(
                core.execution_step('worker:%d' % i, net, num_iter=200))
        super_step = core.execution_step(
            'parent', steps, concurrent_substeps=True)
        plan = core.Plan('plan')
        plan.AddStep(core.execution_step('init', init_net))
        plan.AddStep(super_step)
        workspace.RunPlan(plan)
        # checksum = sum[i=1..20000](i) = 20000 * 20001 / 2 = 200010000
        self.assertEquals(workspace.FetchBlob(checksum), 200010000)

if __name__ == "__main__":
    unittest.main()
