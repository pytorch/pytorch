




from caffe2.python import net_printer
from caffe2.python.checkpoint import Job
from caffe2.python.net_builder import ops
from caffe2.python.task import Task, final_output, WorkspaceType
import unittest


def example_loop():
    with Task():
        total = ops.Const(0)
        total_large = ops.Const(0)
        total_small = ops.Const(0)
        total_tiny = ops.Const(0)
        with ops.loop(10) as loop:
            outer = ops.Mul([loop.iter(), ops.Const(10)])
            with ops.loop(loop.iter()) as inner:
                val = ops.Add([outer, inner.iter()])
                with ops.If(ops.GE([val, ops.Const(80)])) as c:
                    ops.Add([total_large, val], [total_large])
                with c.Elif(ops.GE([val, ops.Const(50)])) as c:
                    ops.Add([total_small, val], [total_small])
                with c.Else():
                    ops.Add([total_tiny, val], [total_tiny])
                ops.Add([total, val], total)


def example_task():
    with Task():
        with ops.task_init():
            one = ops.Const(1)
        two = ops.Add([one, one])
        with ops.task_init():
            three = ops.Const(3)
        accum = ops.Add([two, three])
        # here, accum should be 5
        with ops.task_exit():
            # here, accum should be 6, since this executes after lines below
            seven_1 = ops.Add([accum, one])
        six = ops.Add([accum, one])
        ops.Add([accum, one], [accum])
        seven_2 = ops.Add([accum, one])
        o6 = final_output(six)
        o7_1 = final_output(seven_1)
        o7_2 = final_output(seven_2)

    with Task(num_instances=2):
        with ops.task_init():
            one = ops.Const(1)
        with ops.task_instance_init():
            local = ops.Const(2)
        ops.Add([one, local], [one])
        ops.LogInfo('ble')

    return o6, o7_1, o7_2

def example_job():
    with Job() as job:
        with job.init_group:
            example_loop()
        example_task()
    return job


class TestNetPrinter(unittest.TestCase):
    def test_print(self):
        self.assertTrue(len(net_printer.to_string(example_job())) > 0)

    def test_valid_job(self):
        job = example_job()
        with job:
            with Task():
                # distributed_ctx_init_* ignored by analyzer
                ops.Add(['distributed_ctx_init_a', 'distributed_ctx_init_b'])
        # net_printer.analyze(example_job())
        print(net_printer.to_string(example_job()))

    def test_undefined_blob(self):
        job = example_job()
        with job:
            with Task():
                ops.Add(['a', 'b'])
        with self.assertRaises(AssertionError) as e:
            net_printer.analyze(job)
        self.assertEqual("Blob undefined: a", str(e.exception))

    def test_multiple_definition(self):
        job = example_job()
        with job:
            with Task(workspace_type=WorkspaceType.GLOBAL):
                ops.Add([ops.Const(0), ops.Const(1)], 'out1')
            with Task(workspace_type=WorkspaceType.GLOBAL):
                ops.Add([ops.Const(2), ops.Const(3)], 'out1')
        with self.assertRaises(AssertionError):
            net_printer.analyze(job)
