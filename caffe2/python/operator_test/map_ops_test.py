




import itertools
import numpy as np
import tempfile
import unittest
import os

from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu


class TestMap(hu.HypothesisTestCase):

    def test_create_map(self):
        dtypes = [core.DataType.INT32, core.DataType.INT64]
        for key_dtype, value_dtype in itertools.product(dtypes, dtypes):
            op = core.CreateOperator(
                'CreateMap',
                [],
                ['map'],
                key_dtype=key_dtype,
                value_dtype=value_dtype,
            )
            workspace.RunOperatorOnce(op)
            self.assertTrue(workspace.HasBlob('map'))

    def test_map(self):

        def test_map_func(KEY_T, VALUE_T):
            model_file = os.path.join(tempfile.mkdtemp(), 'db')
            key_data = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=KEY_T)
            value_data = np.asarray([2, 3, 3, 3, 3, 2, 3, 3, 3, 3], dtype=VALUE_T)
            workspace.FeedBlob("key_data", key_data)
            workspace.FeedBlob("value_data", value_data)
            save_net = core.Net("save_net")
            save_net.KeyValueToMap(["key_data", "value_data"], "map_data")
            save_net.Save(
                ["map_data"], [],
                db=model_file,
                db_type="minidb",
                absolute_path=True
            )
            workspace.RunNetOnce(save_net)
            workspace.ResetWorkspace()
            load_net = core.Net("load_net")
            load_net.Load(
                [], ["map_data"],
                db=model_file,
                db_type="minidb",
                load_all=True,
                absolute_path=True
            )
            load_net.MapToKeyValue("map_data", ["key_data", "value_data"])
            workspace.RunNetOnce(load_net)
            key_data2 = workspace.FetchBlob("key_data")
            value_data2 = workspace.FetchBlob("value_data")
            assert(set(zip(key_data, value_data)) == set(zip(key_data2, value_data2)))

        test_map_func(np.int64, np.int64)
        test_map_func(np.int64, np.int32)
        test_map_func(np.int32, np.int32)
        test_map_func(np.int32, np.int64)


if __name__ == "__main__":
    unittest.main()
