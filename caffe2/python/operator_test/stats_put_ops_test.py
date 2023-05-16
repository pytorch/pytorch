from caffe2.python import core, workspace
from caffe2.python.test_util import TestCase
import numpy as np


class TestPutOps(TestCase):
    def test_default_value(self):
        magnitude_expand = int(1e12)
        stat_name = b"stat"
        sum_postfix = b"/stat_value/sum"
        count_postfix = b"/stat_value/count"
        default_value = 16.0

        workspace.FeedBlob("value", np.array([], dtype=np.float))

        workspace.RunOperatorOnce(core.CreateOperator(
            "AveragePut",
            "value",
            [],
            stat_name=stat_name,
            magnitude_expand=magnitude_expand,
            bound=True,
            default_value=default_value))

        workspace.RunOperatorOnce(core.CreateOperator(
            'StatRegistryExport', [], ['k', 'v', 't']))

        k = workspace.FetchBlob('k')
        v = workspace.FetchBlob('v')

        stat_dict = dict(zip(k, v))

        self.assertIn(stat_name + sum_postfix, stat_dict)
        self.assertIn(stat_name + count_postfix, stat_dict)
        self.assertEqual(stat_dict[stat_name + sum_postfix],
         default_value * magnitude_expand)
        self.assertEqual(stat_dict[stat_name + count_postfix], 1)

    def test_clamp(self):
        put_value = 10
        magnitude_expand = int(1e18)
        stat_name = b"stat"
        sum_postfix = b"/stat_value/sum"
        count_postfix = b"/stat_value/count"

        workspace.FeedBlob("value", np.array([put_value], dtype=np.float))

        workspace.RunOperatorOnce(core.CreateOperator(
            "AveragePut",
            "value",
            [],
            stat_name=stat_name,
            magnitude_expand=magnitude_expand,
            bound=True))

        workspace.RunOperatorOnce(core.CreateOperator(
            'StatRegistryExport', [], ['k', 'v', 't']))

        k = workspace.FetchBlob('k')
        v = workspace.FetchBlob('v')

        stat_dict = dict(zip(k, v))

        self.assertIn(stat_name + sum_postfix, stat_dict)
        self.assertIn(stat_name + count_postfix, stat_dict)
        self.assertEqual(stat_dict[stat_name + sum_postfix],
            9223372036854775807)
        self.assertEqual(stat_dict[stat_name + count_postfix], 1)

    def test_clamp_with_out_of_bounds(self):
        put_value = float(1e20)
        magnitude_expand = 1000000000000
        stat_name = b"stat"
        sum_postfix = b"/stat_value/sum"
        count_postfix = b"/stat_value/count"

        workspace.FeedBlob("value", np.array([put_value], dtype=np.float))

        workspace.RunOperatorOnce(core.CreateOperator(
            "AveragePut",
            "value",
            [],
            stat_name=stat_name,
            magnitude_expand=magnitude_expand,
            bound=True))

        workspace.RunOperatorOnce(core.CreateOperator(
            'StatRegistryExport', [], ['k', 'v', 't']))

        k = workspace.FetchBlob('k')
        v = workspace.FetchBlob('v')

        stat_dict = dict(zip(k, v))

        self.assertIn(stat_name + sum_postfix, stat_dict)
        self.assertIn(stat_name + count_postfix, stat_dict)
        self.assertEqual(stat_dict[stat_name + sum_postfix],
            9223372036854775807)
        self.assertEqual(stat_dict[stat_name + count_postfix], 1)

    def test_avg_put_ops(self):
        put_value = 15.1111
        magnitude_expand = 10000
        stat_name = b"a1"
        sum_postfix = b"/stat_value/sum"
        count_postfix = b"/stat_value/count"

        workspace.FeedBlob("value", np.array([put_value], dtype=np.float))

        workspace.RunOperatorOnce(core.CreateOperator(
            "AveragePut",
            "value",
            [],
            stat_name=stat_name,
            magnitude_expand=magnitude_expand))

        workspace.RunOperatorOnce(core.CreateOperator(
            'StatRegistryExport', [], ['k', 'v', 't']))

        k = workspace.FetchBlob('k')
        v = workspace.FetchBlob('v')

        stat_dict = dict(zip(k, v))

        self.assertIn(stat_name + sum_postfix, stat_dict)
        self.assertIn(stat_name + count_postfix, stat_dict)
        self.assertEqual(stat_dict[stat_name + sum_postfix],
         put_value * magnitude_expand)
        self.assertEqual(stat_dict[stat_name + count_postfix], 1)

    def test_increment_put_ops(self):
        put_value = 15.1111
        magnitude_expand = 10000
        stat_name = b"i1"
        member_postfix = b"/stat_value"

        workspace.FeedBlob("value", np.array([put_value], dtype=np.float))

        workspace.RunOperatorOnce(core.CreateOperator(
            "IncrementPut",
            "value",
            [],
            stat_name=stat_name,
            magnitude_expand=magnitude_expand))

        workspace.RunOperatorOnce(core.CreateOperator(
            'StatRegistryExport', [], ['k', 'v', 't']))

        k = workspace.FetchBlob('k')
        v = workspace.FetchBlob('v')

        stat_dict = dict(zip(k, v))

        self.assertIn(stat_name + member_postfix, stat_dict)
        self.assertEqual(stat_dict[stat_name + member_postfix],
         put_value * magnitude_expand)

    def test_stddev_put_ops(self):
        put_value = 15.1111
        magnitude_expand = 10000
        stat_name = b"s1"
        sum_postfix = b"/stat_value/sum"
        count_postfix = b"/stat_value/count"
        sumoffset_postfix = b"/stat_value/sumoffset"
        sumsqoffset_postfix = b"/stat_value/sumsqoffset"

        workspace.FeedBlob("value", np.array([put_value], dtype=np.float))

        workspace.RunOperatorOnce(core.CreateOperator(
            "StdDevPut",
            "value",
            [],
            stat_name=stat_name,
            magnitude_expand=magnitude_expand))

        workspace.RunOperatorOnce(core.CreateOperator(
            'StatRegistryExport', [], ['k', 'v', 't']))

        k = workspace.FetchBlob('k')
        v = workspace.FetchBlob('v')

        stat_dict = dict(zip(k, v))

        self.assertIn(stat_name + sum_postfix, stat_dict)
        self.assertIn(stat_name + count_postfix, stat_dict)
        self.assertIn(stat_name + sumoffset_postfix, stat_dict)
        self.assertIn(stat_name + sumsqoffset_postfix, stat_dict)
        self.assertEqual(stat_dict[stat_name + sum_postfix],
            put_value * magnitude_expand)
        self.assertEqual(stat_dict[stat_name + count_postfix], 1)
