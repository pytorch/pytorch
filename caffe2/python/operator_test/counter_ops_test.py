from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, workspace
from caffe2.python.test_util import TestCase
import tempfile


class TestCounterOps(TestCase):

    def test_counter_ops(self):
        workspace.RunOperatorOnce(core.CreateOperator(
            'CreateCounter', [], ['c'], init_count=1))

        workspace.RunOperatorOnce(core.CreateOperator(
            'CountDown', ['c'], ['t1']))  # 1 -> 0
        assert not workspace.FetchBlob('t1')

        workspace.RunOperatorOnce(core.CreateOperator(
            'CountDown', ['c'], ['t2']))  # 0 -> -1
        assert workspace.FetchBlob('t2')

        workspace.RunOperatorOnce(core.CreateOperator(
            'CountUp', ['c'], ['t21']))  # -1 -> 0
        assert workspace.FetchBlob('t21') == -1
        workspace.RunOperatorOnce(core.CreateOperator(
            'RetrieveCount', ['c'], ['t22']))
        assert workspace.FetchBlob('t22') == 0

        workspace.RunOperatorOnce(core.CreateOperator(
            'ResetCounter', ['c'], [], init_count=1))  # -> 1
        workspace.RunOperatorOnce(core.CreateOperator(
            'CountDown', ['c'], ['t3']))  # 1 -> 0
        assert not workspace.FetchBlob('t3')

        workspace.RunOperatorOnce(core.CreateOperator(
            'ResetCounter', ['c'], ['t31'], init_count=5))  # 0 -> 5
        assert workspace.FetchBlob('t31') == 0
        workspace.RunOperatorOnce(core.CreateOperator(
            'ResetCounter', ['c'], ['t32']))  # 5 -> 0
        assert workspace.FetchBlob('t32') == 5

        workspace.RunOperatorOnce(core.CreateOperator(
            'ConstantFill', [], ['t4'], value=False, shape=[],
            dtype=core.DataType.BOOL))
        assert workspace.FetchBlob('t4') == workspace.FetchBlob('t1')

        workspace.RunOperatorOnce(core.CreateOperator(
            'ConstantFill', [], ['t5'], value=True, shape=[],
            dtype=core.DataType.BOOL))
        assert workspace.FetchBlob('t5') == workspace.FetchBlob('t2')

        assert workspace.RunOperatorOnce(core.CreateOperator(
            'And', ['t1', 't2'], ['t6']))
        assert not workspace.FetchBlob('t6')  # True && False

        assert workspace.RunOperatorOnce(core.CreateOperator(
            'And', ['t2', 't5'], ['t7']))
        assert workspace.FetchBlob('t7')  # True && True

        workspace.RunOperatorOnce(core.CreateOperator(
            'CreateCounter', [], ['serialized_c'], init_count=22))
        with tempfile.NamedTemporaryFile() as tmp:
            workspace.RunOperatorOnce(core.CreateOperator(
                'Save', ['serialized_c'], [], absolute_path=1,
                db_type='minidb', db=tmp.name))
            for i in range(10):
                workspace.RunOperatorOnce(core.CreateOperator(
                    'CountDown', ['serialized_c'], ['t8']))
            workspace.RunOperatorOnce(core.CreateOperator(
                'RetrieveCount', ['serialized_c'], ['t8']))
            assert workspace.FetchBlob('t8') == 12
            workspace.RunOperatorOnce(core.CreateOperator(
                'Load', [], ['serialized_c'], absolute_path=1,
                db_type='minidb', db=tmp.name))
            workspace.RunOperatorOnce(core.CreateOperator(
                'RetrieveCount', ['serialized_c'], ['t8']))
            assert workspace.FetchBlob('t8') == 22

if __name__ == "__main__":
    import unittest
    unittest.main()
