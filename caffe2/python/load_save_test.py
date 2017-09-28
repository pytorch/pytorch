# Copyright (c) 2016-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import errno
import hypothesis.strategies as st
from hypothesis import given
import numpy as np
import os
import shutil
import tempfile
import unittest

from caffe2.proto import caffe2_pb2
from caffe2.python import core, test_util, workspace

if workspace.has_gpu_support:
    DEVICES = [caffe2_pb2.CPU, caffe2_pb2.CUDA]
    max_gpuid = workspace.NumCudaDevices() - 1
else:
    DEVICES = [caffe2_pb2.CPU]
    max_gpuid = 0


# Utility class for other loading tests, don't add test functions here
# Inherit from this test instead. If you add a test here,
# each derived class will inherit it as well and cause test duplication
class TestLoadSaveBase(test_util.TestCase):

    def __init__(self, methodName, db_type='minidb'):
        super(TestLoadSaveBase, self).__init__(methodName)
        self._db_type = db_type

    @given(src_device_type=st.sampled_from(DEVICES),
           src_gpu_id=st.integers(min_value=0, max_value=max_gpuid),
           dst_device_type=st.sampled_from(DEVICES),
           dst_gpu_id=st.integers(min_value=0, max_value=max_gpuid))
    def load_save(self, src_device_type, src_gpu_id,
                  dst_device_type, dst_gpu_id):
        workspace.ResetWorkspace()
        dtypes = [np.float16, np.float32, np.float64, np.bool, np.int8,
                  np.int16, np.int32, np.int64, np.uint8, np.uint16]
        arrays = [np.random.permutation(6).reshape(2, 3).astype(T)
                  for T in dtypes]
        src_device_option = core.DeviceOption(
            src_device_type, src_gpu_id)
        dst_device_option = core.DeviceOption(
            dst_device_type, dst_gpu_id)

        for i, arr in enumerate(arrays):
            self.assertTrue(workspace.FeedBlob(str(i), arr, src_device_option))
            self.assertTrue(workspace.HasBlob(str(i)))

        try:
            # Saves the blobs to a local db.
            tmp_folder = tempfile.mkdtemp()
            op = core.CreateOperator(
                "Save",
                [str(i) for i in range(len(arrays))], [],
                absolute_path=1,
                db=os.path.join(tmp_folder, "db"), db_type=self._db_type)
            self.assertTrue(workspace.RunOperatorOnce(op))

            # Reset the workspace so that anything we load is surely loaded
            # from the serialized proto.
            workspace.ResetWorkspace()
            self.assertEqual(len(workspace.Blobs()), 0)

            def _LoadTest(keep_device, device_type, gpu_id, blobs, loadAll):
                """A helper subfunction to test keep and not keep."""
                op = core.CreateOperator(
                    "Load",
                    [], blobs,
                    absolute_path=1,
                    db=os.path.join(tmp_folder, "db"), db_type=self._db_type,
                    device_option=dst_device_option,
                    keep_device=keep_device,
                    load_all=loadAll)
                self.assertTrue(workspace.RunOperatorOnce(op))
                for i, arr in enumerate(arrays):
                    self.assertTrue(workspace.HasBlob(str(i)))
                    fetched = workspace.FetchBlob(str(i))
                    self.assertEqual(fetched.dtype, arr.dtype)
                    np.testing.assert_array_equal(
                        workspace.FetchBlob(str(i)), arr)
                    proto = caffe2_pb2.BlobProto()
                    proto.ParseFromString(workspace.SerializeBlob(str(i)))
                    self.assertTrue(proto.HasField('tensor'))
                    self.assertEqual(proto.tensor.device_detail.device_type,
                                     device_type)
                    if device_type == caffe2_pb2.CUDA:
                        self.assertEqual(proto.tensor.device_detail.cuda_gpu_id,
                                         gpu_id)

            blobs = [str(i) for i in range(len(arrays))]
            # Load using device option stored in the proto, i.e.
            # src_device_option
            _LoadTest(1, src_device_type, src_gpu_id, blobs, 0)
            # Load again, but this time load into dst_device_option.
            _LoadTest(0, dst_device_type, dst_gpu_id, blobs, 0)
            # Load back to the src_device_option to see if both paths are able
            # to reallocate memory.
            _LoadTest(1, src_device_type, src_gpu_id, blobs, 0)
            # Reset the workspace, and load directly into the dst_device_option.
            workspace.ResetWorkspace()
            _LoadTest(0, dst_device_type, dst_gpu_id, blobs, 0)

            # Test load all which loads all blobs in the db into the workspace.
            workspace.ResetWorkspace()
            _LoadTest(1, src_device_type, src_gpu_id, [], 1)
            # Load again making sure that overwrite functionality works.
            _LoadTest(1, src_device_type, src_gpu_id, [], 1)
            # Load again with different device.
            _LoadTest(0, dst_device_type, dst_gpu_id, [], 1)
            workspace.ResetWorkspace()
            _LoadTest(0, dst_device_type, dst_gpu_id, [], 1)
        finally:
            # clean up temp folder.
            try:
                shutil.rmtree(tmp_folder)
            except OSError as e:
                if e.errno != errno.ENOENT:
                    raise

    def saveFile(self, tmp_folder, db_type):
        dtypes = [np.float16, np.float32, np.float64, np.bool, np.int8,
                  np.int16, np.int32, np.int64, np.uint8, np.uint16]
        arrays = [np.random.permutation(6).reshape(2, 3).astype(T)
                  for T in dtypes]

        for i, arr in enumerate(arrays):
            self.assertTrue(workspace.FeedBlob(str(i), arr))
            self.assertTrue(workspace.HasBlob(str(i)))

        # Saves the blobs to a local db.
        tmp_file = os.path.join(tmp_folder, "db")
        op = core.CreateOperator(
            "Save",
            [str(i) for i in range(len(arrays))], [],
            absolute_path=1,
            db=tmp_file, db_type=db_type)
        workspace.RunOperatorOnce(op)
        return tmp_file, arrays


class TestLoadSave(TestLoadSaveBase):

    def testLoadSave(self):
        self.load_save()

    def testRepeatedArgs(self):
        dtypes = [np.float16, np.float32, np.float64, np.bool, np.int8,
                  np.int16, np.int32, np.int64, np.uint8, np.uint16]
        arrays = [np.random.permutation(6).reshape(2, 3).astype(T)
                  for T in dtypes]

        for i, arr in enumerate(arrays):
            self.assertTrue(workspace.FeedBlob(str(i), arr))
            self.assertTrue(workspace.HasBlob(str(i)))

        # Saves the blobs to a local db.
        tmp_folder = tempfile.mkdtemp()
        op = core.CreateOperator(
            "Save",
            [str(i) for i in range(len(arrays))] * 2, [],
            absolute_path=1,
            db=os.path.join(tmp_folder, "db"), db_type=self._db_type)
        with self.assertRaises(RuntimeError):
            workspace.RunOperatorOnce(op)
        try:
            shutil.rmtree(tmp_folder)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise

    def testLoadExcessblobs(self):
        tmp_folder = tempfile.mkdtemp()
        tmp_file, arrays = self.saveFile(tmp_folder, self._db_type)

        op = core.CreateOperator(
            "Load",
            [], [str(i) for i in range(len(arrays))] * 2,
            absolute_path=1,
            db=tmp_file, db_type=self._db_type,
            load_all=False)
        with self.assertRaises(RuntimeError):
            workspace.RunOperatorOnce(op)

        try:
            shutil.rmtree(tmp_folder)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise

    def testTruncatedFile(self):
        tmp_folder = tempfile.mkdtemp()
        tmp_file, arrays = self.saveFile(tmp_folder, self._db_type)

        with open(tmp_file, 'wb+') as fdest:
            fdest.seek(20, os.SEEK_END)
            fdest.truncate()

        op = core.CreateOperator(
            "Load",
            [], [str(i) for i in range(len(arrays))],
            absolute_path=1,
            db=tmp_file, db_type=self._db_type,
            load_all=False)
        with self.assertRaises(RuntimeError):
            workspace.RunOperatorOnce(op)

        op = core.CreateOperator(
            "Load",
            [], [],
            absolute_path=1,
            db=tmp_file, db_type=self._db_type,
            load_all=True)
        with self.assertRaises(RuntimeError):
            workspace.RunOperatorOnce(op)
        try:
            shutil.rmtree(tmp_folder)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise

    def testBlobNameOverrides(self):
        original_names = ['blob_a', 'blob_b', 'blob_c']
        new_names = ['x', 'y', 'z']
        blobs = [np.random.permutation(6) for i in range(3)]
        for i, blob in enumerate(blobs):
            self.assertTrue(workspace.FeedBlob(original_names[i], blob))
            self.assertTrue(workspace.HasBlob(original_names[i]))
        self.assertEqual(len(workspace.Blobs()), 3)

        try:
            # Saves the blobs to a local db.
            tmp_folder = tempfile.mkdtemp()
            with self.assertRaises(RuntimeError):
                workspace.RunOperatorOnce(
                    core.CreateOperator(
                        "Save", original_names, [],
                        absolute_path=1,
                        strip_prefix='.temp',
                        blob_name_overrides=new_names,
                        db=os.path.join(tmp_folder, "db"),
                        db_type=self._db_type
                    )
                )
            self.assertTrue(
                workspace.RunOperatorOnce(
                    core.CreateOperator(
                        "Save", original_names, [],
                        absolute_path=1,
                        blob_name_overrides=new_names,
                        db=os.path.join(tmp_folder, "db"),
                        db_type=self._db_type
                    )
                )
            )
            self.assertTrue(workspace.ResetWorkspace())
            self.assertEqual(len(workspace.Blobs()), 0)
            self.assertTrue(
                workspace.RunOperatorOnce(
                    core.CreateOperator(
                        "Load", [], [],
                        absolute_path=1,
                        db=os.path.join(tmp_folder, "db"),
                        db_type=self._db_type,
                        load_all=1
                    )
                )
            )
            self.assertEqual(len(workspace.Blobs()), 3)
            for i, name in enumerate(new_names):
                self.assertTrue(workspace.HasBlob(name))
                self.assertTrue((workspace.FetchBlob(name) == blobs[i]).all())
            # moved here per @cxj's suggestion
            load_new_names = ['blob_x', 'blob_y', 'blob_z']
            # load 'x' into 'blob_x'
            self.assertTrue(
                workspace.RunOperatorOnce(
                    core.CreateOperator(
                        "Load", [], load_new_names[0:1],
                        absolute_path=1,
                        db=os.path.join(tmp_folder, "db"),
                        db_type=self._db_type,
                        source_blob_names=new_names[0:1]
                    )
                )
            )
            # we should have 'blob_a/b/c/' and 'blob_x' now
            self.assertEqual(len(workspace.Blobs()), 4)
            for i, name in enumerate(load_new_names[0:1]):
                self.assertTrue(workspace.HasBlob(name))
                self.assertTrue((workspace.FetchBlob(name) == blobs[i]).all())
            self.assertTrue(
                workspace.RunOperatorOnce(
                    core.CreateOperator(
                        "Load", [], load_new_names[0:3],
                        absolute_path=1,
                        db=os.path.join(tmp_folder, "db"),
                        db_type=self._db_type,
                        source_blob_names=new_names[0:3]
                    )
                )
            )
            # we should have 'blob_a/b/c/' and 'blob_x/y/z' now
            self.assertEqual(len(workspace.Blobs()), 6)
            for i, name in enumerate(load_new_names[0:3]):
                self.assertTrue(workspace.HasBlob(name))
                self.assertTrue((workspace.FetchBlob(name) == blobs[i]).all())
        finally:
            # clean up temp folder.
            try:
                shutil.rmtree(tmp_folder)
            except OSError as e:
                if e.errno != errno.ENOENT:
                    raise

    def testMissingFile(self):
        tmp_folder = tempfile.mkdtemp()
        tmp_file = os.path.join(tmp_folder, "missing_db")

        op = core.CreateOperator(
            "Load",
            [], [],
            absolute_path=1,
            db=tmp_file, db_type=self._db_type,
            load_all=True)
        with self.assertRaises(RuntimeError):
            try:
                workspace.RunOperatorOnce(op)
            except RuntimeError as e:
                print(e)
                raise
        try:
            shutil.rmtree(tmp_folder)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise



if __name__ == '__main__':
    unittest.main()
