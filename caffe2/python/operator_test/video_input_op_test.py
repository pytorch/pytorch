from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
try:
    import lmdb
except ImportError:
    raise unittest.SkipTest('python-lmdb is not installed')

import sys
import os
import shutil
import tempfile

from caffe2.proto import caffe2_pb2
from caffe2.python import workspace, model_helper
import numpy as np


class VideoInputOpTest(unittest.TestCase):

    def create_a_list(self, output_file, line, n):
        # create a list that repeat a line n times
        # used for creating a list file for simple test input
        with open(output_file, 'w') as file:
            for _i in range(n):
                file.write(line)

    def create_video_db(self, list_file, output_file, use_list=False):
        # Write to lmdb database...
        LMDB_MAP_SIZE = 1 << 40   # MODIFY
        env = lmdb.open(output_file, map_size=LMDB_MAP_SIZE)
        total_size = 0

        file_name = []
        start_frame = []
        label = []
        index = 0

        with env.begin(write=True) as txn:
            with open(list_file, 'r') as data:
                for line in data:
                    p = line.split()
                    file_name = p[0]
                    start_frame = int(p[1])
                    label = int(p[2])

                    if not use_list:
                        with open(file_name, mode='rb') as file:
                            video_data = file.read()
                    else:
                        video_data = file_name

                    tensor_protos = caffe2_pb2.TensorProtos()
                    video_tensor = tensor_protos.protos.add()
                    video_tensor.data_type = 4  # string data
                    video_tensor.string_data.append(video_data)

                    label_tensor = tensor_protos.protos.add()
                    label_tensor.data_type = 2
                    label_tensor.int32_data.append(label)

                    start_frame_tensor = tensor_protos.protos.add()
                    start_frame_tensor.data_type = 2
                    start_frame_tensor.int32_data.append(start_frame)

                    txn.put(
                        '{}'.format(index).encode('ascii'),
                        tensor_protos.SerializeToString()
                    )
                    index = index + 1
                    total_size = total_size + len(video_data) + sys.getsizeof(int)

        return total_size

    def test_read_from_db(self):
        random_label = np.random.randint(0, 100)
        VIDEO = "/mnt/vol/gfsdataswarm-oregon/users/trandu/sample.avi"
        if not os.path.exists(VIDEO):
            raise unittest.SkipTest('Missing data')
        temp_list = tempfile.NamedTemporaryFile(delete=False).name
        line_str = '{} 0 {}\n'.format(VIDEO, random_label)
        self.create_a_list(
            temp_list,
            line_str,
            16)
        video_db_dir = tempfile.mkdtemp()

        self.create_video_db(temp_list, video_db_dir)
        model = model_helper.ModelHelper(name="Video Loader from LMDB")
        reader = model.CreateDB(
            "sample",
            db=video_db_dir,
            db_type="lmdb")
        model.VideoInput(
            reader,
            ["data", "label"],
            name="data",
            batch_size=10,
            width=171,
            height=128,
            crop=112,
            length=8,
            sampling_rate=2,
            mirror=1,
            use_local_file=0,
            temporal_jitter=1)

        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)
        data = workspace.FetchBlob("data")
        label = workspace.FetchBlob("label")

        np.testing.assert_equal(label, random_label)
        np.testing.assert_equal(data.shape, [10, 3, 8, 112, 112])
        os.remove(temp_list)
        shutil.rmtree(video_db_dir)


if __name__ == "__main__":
    unittest.main()
