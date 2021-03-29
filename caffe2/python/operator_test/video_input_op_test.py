

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
from caffe2.proto import caffe2_pb2
from caffe2.python import model_helper, workspace


try:
    import lmdb
except ImportError:
    raise unittest.SkipTest("python-lmdb is not installed")


class VideoInputOpTest(unittest.TestCase):
    def create_a_list(self, output_file, line, n):
        # create a list that repeat a line n times
        # used for creating a list file for simple test input
        with open(output_file, "w") as file:
            for _i in range(n):
                file.write(line)

    def create_video_db(self, list_file, output_file, use_list=False):
        # Write to lmdb database...
        LMDB_MAP_SIZE = 1 << 40  # MODIFY
        env = lmdb.open(output_file, map_size=LMDB_MAP_SIZE)
        total_size = 0

        file_name = []
        start_frame = []
        label = []
        index = 0

        with env.begin(write=True) as txn:
            with open(list_file, "r") as data:
                for line in data:
                    p = line.split()
                    file_name = p[0]
                    start_frame = int(p[1])
                    label = int(p[2])

                    if not use_list:
                        with open(file_name, mode="rb") as file:
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
                        "{}".format(index).encode("ascii"),
                        tensor_protos.SerializeToString(),
                    )
                    index = index + 1
                    total_size = total_size + len(video_data) + sys.getsizeof(int)
        return total_size

    # sample one clip randomly from the video
    def test_rgb_with_temporal_jittering(self):
        random_label = np.random.randint(0, 100)
        VIDEO = "/mnt/vol/gfsdataswarm-oregon/users/trandu/sample.avi"
        if not os.path.exists(VIDEO):
            raise unittest.SkipTest("Missing data")
        temp_list = tempfile.NamedTemporaryFile(delete=False).name
        line_str = "{} 0 {}\n".format(VIDEO, random_label)
        self.create_a_list(temp_list, line_str, 16)
        video_db_dir = tempfile.mkdtemp()

        self.create_video_db(temp_list, video_db_dir)
        model = model_helper.ModelHelper(name="Video Loader from LMDB")
        reader = model.CreateDB("sample", db=video_db_dir, db_type="lmdb")

        # build the model
        model.net.VideoInput(
            reader,
            ["data", "label"],
            name="data",
            batch_size=16,
            clip_per_video=1,
            crop_size=112,
            scale_w=171,
            scale_h=128,
            length_rgb=8,
            sampling_rate_rgb=1,
            decode_type=0,
            video_res_type=0,  # scale by scale_h and scale_w
        )

        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)
        data = workspace.FetchBlob("data")
        label = workspace.FetchBlob("label")

        np.testing.assert_equal(label, random_label)
        np.testing.assert_equal(data.shape, [16, 3, 8, 112, 112])
        os.remove(temp_list)
        shutil.rmtree(video_db_dir)

    # sample multiple clips uniformly from the video
    def test_rgb_with_uniform_sampling(self):
        random_label = np.random.randint(0, 100)
        clip_per_video = np.random.randint(2, 11)
        VIDEO = "/mnt/vol/gfsdataswarm-oregon/users/trandu/sample.avi"
        if not os.path.exists(VIDEO):
            raise unittest.SkipTest("Missing data")
        temp_list = tempfile.NamedTemporaryFile(delete=False).name
        line_str = "{} 0 {}\n".format(VIDEO, random_label)
        self.create_a_list(temp_list, line_str, 16)
        video_db_dir = tempfile.mkdtemp()

        self.create_video_db(temp_list, video_db_dir)
        model = model_helper.ModelHelper(name="Video Loader from LMDB")
        reader = model.CreateDB("sample", db=video_db_dir, db_type="lmdb")

        # build the model
        model.net.VideoInput(
            reader,
            ["data", "label"],
            name="data",
            batch_size=3,
            clip_per_video=clip_per_video,
            crop_size=112,
            scale_w=171,
            scale_h=128,
            length_rgb=8,
            sampling_rate_rgb=1,
            decode_type=1,
            video_res_type=0,
        )

        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)
        data = workspace.FetchBlob("data")
        label = workspace.FetchBlob("label")

        np.testing.assert_equal(label, random_label)
        np.testing.assert_equal(data.shape, [3 * clip_per_video, 3, 8, 112, 112])
        os.remove(temp_list)
        shutil.rmtree(video_db_dir)

    # test optical flow
    def test_optical_flow_with_temporal_jittering(self):
        random_label = np.random.randint(0, 100)
        VIDEO = "/mnt/vol/gfsdataswarm-oregon/users/trandu/sample.avi"
        if not os.path.exists(VIDEO):
            raise unittest.SkipTest("Missing data")
        temp_list = tempfile.NamedTemporaryFile(delete=False).name
        line_str = "{} 0 {}\n".format(VIDEO, random_label)
        self.create_a_list(temp_list, line_str, 16)
        video_db_dir = tempfile.mkdtemp()

        self.create_video_db(temp_list, video_db_dir)
        model = model_helper.ModelHelper(name="Video Loader from LMDB")
        reader = model.CreateDB("sample", db=video_db_dir, db_type="lmdb")
        model.net.VideoInput(
            reader,
            ["data", "label"],
            name="data",
            batch_size=16,
            clip_per_video=1,
            crop_size=112,
            scale_w=171,
            scale_h=128,
            length_of=8,
            sampling_rate_of=1,
            frame_gap_of=1,
            decode_type=0,
            video_res_type=0,
            get_rgb=False,
            get_optical_flow=True,
        )

        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)
        data = workspace.FetchBlob("data")
        label = workspace.FetchBlob("label")

        np.testing.assert_equal(label, random_label)
        np.testing.assert_equal(data.shape, [16, 2, 8, 112, 112])
        os.remove(temp_list)
        shutil.rmtree(video_db_dir)

    # test rgb output VideoResType is
    # USE_SHORTER_EDGE
    def test_rgb_use_shorter_edge(self):
        batch_size = 16
        random_label = np.random.randint(0, 100)
        VIDEO = "/mnt/vol/gfsdataswarm-oregon/users/trandu/sample.avi"
        if not os.path.exists(VIDEO):
            raise unittest.SkipTest("Missing data")
        temp_list = tempfile.NamedTemporaryFile(delete=False).name
        line_str = "{} 0 {}\n".format(VIDEO, random_label)
        self.create_a_list(temp_list, line_str, batch_size)
        video_db_dir = tempfile.mkdtemp()

        self.create_video_db(temp_list, video_db_dir)
        model = model_helper.ModelHelper(name="Video Loader from LMDB")
        reader = model.CreateDB("sample", db=video_db_dir, db_type="lmdb")
        model.net.VideoInput(
            reader,
            ["data", "label"],
            name="data",
            batch_size=batch_size,
            clip_per_video=1,
            crop_size=112,
            scale_w=171,
            scale_h=128,
            length_of=8,
            frame_gap_of=1,
            decode_type=0,
            video_res_type=1,  # use shorter edge
            get_rgb=True,
            length_rgb=8,
            short_edge=112,
        )

        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)
        data = workspace.FetchBlob("data")
        label = workspace.FetchBlob("label")

        np.testing.assert_equal(label.shape, [batch_size])
        for i in range(batch_size):
            np.testing.assert_equal(label[i], random_label)
        np.testing.assert_equal(data.shape, [batch_size, 3, 8, 112, 112])
        os.remove(temp_list)
        shutil.rmtree(video_db_dir)

    # test optical flow output VideoResType is
    # USE_SHORTER_EDGE
    def test_optical_flow_use_shorter_edge(self):
        batch_size = 16
        random_label = np.random.randint(0, 100)
        VIDEO = "/mnt/vol/gfsdataswarm-oregon/users/trandu/sample.avi"
        if not os.path.exists(VIDEO):
            raise unittest.SkipTest("Missing data")
        temp_list = tempfile.NamedTemporaryFile(delete=False).name
        line_str = "{} 0 {}\n".format(VIDEO, random_label)
        self.create_a_list(temp_list, line_str, batch_size)
        video_db_dir = tempfile.mkdtemp()

        self.create_video_db(temp_list, video_db_dir)
        model = model_helper.ModelHelper(name="Video Loader from LMDB")
        reader = model.CreateDB("sample", db=video_db_dir, db_type="lmdb")
        model.net.VideoInput(
            reader,
            ["data", "label"],
            name="data",
            batch_size=batch_size,
            clip_per_video=1,
            crop_size=112,
            scale_w=171,
            scale_h=128,
            length_of=8,
            sampling_rate_of=1,
            frame_gap_of=1,
            decode_type=0,
            video_res_type=1,  # use shorter edge
            get_rgb=False,
            get_optical_flow=True,
            short_edge=112,
        )

        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)
        data = workspace.FetchBlob("data")
        label = workspace.FetchBlob("label")

        np.testing.assert_equal(label.shape, [batch_size])
        for i in range(batch_size):
            np.testing.assert_equal(label[i], random_label)
        np.testing.assert_equal(data.shape, [batch_size, 2, 8, 112, 112])
        os.remove(temp_list)
        shutil.rmtree(video_db_dir)


if __name__ == "__main__":
    unittest.main()
