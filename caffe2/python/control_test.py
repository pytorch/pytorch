from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import control, core, test_util, workspace

import logging
logger = logging.getLogger(__name__)


class TestControl(test_util.TestCase):
    def setUp(self):
        super(TestControl, self).setUp()
        self.N_ = 10

        self.init_net_ = core.Net("init-net")
        cnt = self.init_net_.CreateCounter([], init_count=0)
        const_n = self.init_net_.ConstantFill(
            [], shape=[], value=self.N_, dtype=core.DataType.INT64)
        const_0 = self.init_net_.ConstantFill(
            [], shape=[], value=0, dtype=core.DataType.INT64)

        self.cnt_net_ = core.Net("cnt-net")
        self.cnt_net_.CountUp([cnt])
        curr_cnt = self.cnt_net_.RetrieveCount([cnt])
        self.init_net_.ConstantFill(
            [], [curr_cnt], shape=[], value=0, dtype=core.DataType.INT64)
        self.cnt_net_.AddExternalOutput(curr_cnt)

        self.cond_net_ = core.Net("cond-net")
        cond_blob = self.cond_net_.LT([curr_cnt, const_n])
        self.cond_net_.AddExternalOutput(cond_blob)

        self.not_cond_net_ = core.Net("not-cond-net")
        cond_blob = self.not_cond_net_.GE([curr_cnt, const_n])
        self.not_cond_net_.AddExternalOutput(cond_blob)

        self.true_cond_net_ = core.Net("true-cond-net")
        true_blob = self.true_cond_net_.LT([const_0, const_n])
        self.true_cond_net_.AddExternalOutput(true_blob)

        self.false_cond_net_ = core.Net("false-cond-net")
        false_blob = self.false_cond_net_.GT([const_0, const_n])
        self.false_cond_net_.AddExternalOutput(false_blob)

    def CheckNetOutput(self, nets_and_expects):
        """
        Check the net output is expected
        nets_and_expects is a list of tuples (net, expect)
        """
        for net, expect in nets_and_expects:
            output = workspace.FetchBlob(
                net.Proto().external_output[-1])
            self.assertEqual(output, expect)

    def BuildAndRunPlan(self, step):
        plan = core.Plan("test")
        plan.AddStep(control.Do(self.init_net_))
        plan.AddStep(step)
        self.assertEqual(workspace.RunPlan(plan), True)

    def ForLoopTest(self, net_or_step):
        step = control.For(net_or_step, self.N_)
        self.BuildAndRunPlan(step)
        self.CheckNetOutput([(self.cnt_net_, self.N_)])

    def testForLoopWithNet(self):
        self.ForLoopTest(self.cnt_net_)

    def testForLoopWithStep(self):
        step = control.Do(self.cnt_net_)
        self.ForLoopTest(step)

    def WhileLoopTest(self, net_or_step):
        step = control.While(self.cond_net_, net_or_step)
        self.BuildAndRunPlan(step)
        self.CheckNetOutput([(self.cnt_net_, self.N_)])

    def testWhileLoopWithNet(self):
        self.WhileLoopTest(self.cnt_net_)

    def testWhileLoopWithStep(self):
        step = control.Do(self.cnt_net_)
        self.WhileLoopTest(step)

    def UntilLoopTest(self, net_or_step):
        step = control.Until(self.not_cond_net_, net_or_step)
        self.BuildAndRunPlan(step)
        self.CheckNetOutput([(self.cnt_net_, self.N_)])

    def testUntilLoopWithNet(self):
        self.UntilLoopTest(self.cnt_net_)

    def testUntilLoopWithStep(self):
        step = control.Do(self.cnt_net_)
        self.UntilLoopTest(step)

    def DoWhileLoopTest(self, net_or_step):
        step = control.DoWhile(self.cond_net_, net_or_step)
        self.BuildAndRunPlan(step)
        self.CheckNetOutput([(self.cnt_net_, self.N_)])

    def testDoWhileLoopWithNet(self):
        self.DoWhileLoopTest(self.cnt_net_)

    def testDoWhileLoopWithStep(self):
        step = control.Do(self.cnt_net_)
        self.DoWhileLoopTest(step)

    def DoUntilLoopTest(self, net_or_step):
        step = control.DoUntil(self.not_cond_net_, net_or_step)
        self.BuildAndRunPlan(step)
        self.CheckNetOutput([(self.cnt_net_, self.N_)])

    def testDoUntilLoopWithNet(self):
        self.DoUntilLoopTest(self.cnt_net_)

    def testDoUntilLoopWithStep(self):
        step = control.Do(self.cnt_net_)
        self.DoUntilLoopTest(step)

    def IfCondTest(self, cond_net, expect, cond_on_blob):
        if cond_on_blob:
            step = control.Do(
                control.Do(cond_net),
                control.If(cond_net.Proto().external_output[-1],
                           self.cnt_net_))
        else:
            step = control.If(cond_net, self.cnt_net_)
        self.BuildAndRunPlan(step)
        self.CheckNetOutput([(self.cnt_net_, expect)])

    def testIfCondTrueOnNet(self):
        self.IfCondTest(self.true_cond_net_, 1, False)

    def testIfCondTrueOnBlob(self):
        self.IfCondTest(self.true_cond_net_, 1, True)

    def testIfCondFalseOnNet(self):
        self.IfCondTest(self.false_cond_net_, 0, False)

    def testIfCondFalseOnBlob(self):
        self.IfCondTest(self.false_cond_net_, 0, True)

    def IfElseCondTest(self, cond_net, expect, cond_on_blob):
        true_step = control.For(self.cnt_net_, self.N_)
        false_step = control.For(self.cnt_net_, 2 * self.N_)
        if cond_on_blob:
            step = control.Do(
                control.Do(cond_net),
                control.If(cond_net.Proto().external_output[-1],
                           true_step, false_step))
        else:
            step = control.If(cond_net, true_step, false_step)
        self.BuildAndRunPlan(step)
        self.CheckNetOutput([(self.cnt_net_, expect)])

    def testIfElseCondTrueOnNet(self):
        self.IfElseCondTest(self.true_cond_net_, self.N_, False)

    def testIfElseCondTrueOnBlob(self):
        self.IfElseCondTest(self.true_cond_net_, self.N_, True)

    def testIfElseCondFalseOnNet(self):
        self.IfElseCondTest(self.false_cond_net_, 2 * self.N_, False)

    def testIfElseCondFalseOnBlob(self):
        self.IfElseCondTest(self.false_cond_net_, 2 * self.N_, True)

    def IfNotCondTest(self, cond_net, expect, cond_on_blob):
        if cond_on_blob:
            step = control.Do(
                control.Do(cond_net),
                control.IfNot(cond_net.Proto().external_output[-1],
                              self.cnt_net_))
        else:
            step = control.IfNot(cond_net, self.cnt_net_)
        self.BuildAndRunPlan(step)
        self.CheckNetOutput([(self.cnt_net_, expect)])

    def testIfNotCondTrueOnNet(self):
        self.IfNotCondTest(self.true_cond_net_, 0, False)

    def testIfNotCondTrueOnBlob(self):
        self.IfNotCondTest(self.true_cond_net_, 0, True)

    def testIfNotCondFalseOnNet(self):
        self.IfNotCondTest(self.false_cond_net_, 1, False)

    def testIfNotCondFalseOnBlob(self):
        self.IfNotCondTest(self.false_cond_net_, 1, True)

    def IfNotElseCondTest(self, cond_net, expect, cond_on_blob):
        true_step = control.For(self.cnt_net_, self.N_)
        false_step = control.For(self.cnt_net_, 2 * self.N_)
        if cond_on_blob:
            step = control.Do(
                control.Do(cond_net),
                control.IfNot(cond_net.Proto().external_output[-1],
                              true_step, false_step))
        else:
            step = control.IfNot(cond_net, true_step, false_step)
        self.BuildAndRunPlan(step)
        self.CheckNetOutput([(self.cnt_net_, expect)])

    def testIfNotElseCondTrueOnNet(self):
        self.IfNotElseCondTest(self.true_cond_net_, 2 * self.N_, False)

    def testIfNotElseCondTrueOnBlob(self):
        self.IfNotElseCondTest(self.true_cond_net_, 2 * self.N_, True)

    def testIfNotElseCondFalseOnNet(self):
        self.IfNotElseCondTest(self.false_cond_net_, self.N_, False)

    def testIfNotElseCondFalseOnBlob(self):
        self.IfNotElseCondTest(self.false_cond_net_, self.N_, True)
