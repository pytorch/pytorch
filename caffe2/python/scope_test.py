from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import scope, core
from caffe2.proto import caffe2_pb2

import unittest
import threading
import time

SUCCESS_COUNT = 0


def thread_runner(idx, testobj):
    global SUCCESS_COUNT
    testobj.assertEquals(scope.CurrentNameScope(), "")
    testobj.assertEquals(scope.CurrentDeviceScope(), None)
    namescope = "namescope_{}".format(idx)
    dsc = core.DeviceOption(caffe2_pb2.CUDA, idx)
    with scope.DeviceScope(dsc):
        with scope.NameScope(namescope):
            testobj.assertEquals(scope.CurrentNameScope(), namescope + "/")
            testobj.assertEquals(scope.CurrentDeviceScope(), dsc)

            time.sleep(0.01 + idx * 0.01)
            testobj.assertEquals(scope.CurrentNameScope(), namescope + "/")
            testobj.assertEquals(scope.CurrentDeviceScope(), dsc)

    testobj.assertEquals(scope.CurrentNameScope(), "")
    testobj.assertEquals(scope.CurrentDeviceScope(), None)
    SUCCESS_COUNT += 1


class TestScope(unittest.TestCase):

    def testNamescopeBasic(self):
        self.assertEquals(scope.CurrentNameScope(), "")

        with scope.NameScope("test_scope"):
            self.assertEquals(scope.CurrentNameScope(), "test_scope/")

        self.assertEquals(scope.CurrentNameScope(), "")

    def testNamescopeAssertion(self):
        self.assertEquals(scope.CurrentNameScope(), "")

        try:
            with scope.NameScope("test_scope"):
                self.assertEquals(scope.CurrentNameScope(), "test_scope/")
                raise Exception()
        except Exception:
            pass

        self.assertEquals(scope.CurrentNameScope(), "")

    def testDevicescopeBasic(self):
        self.assertEquals(scope.CurrentDeviceScope(), None)

        dsc = core.DeviceOption(caffe2_pb2.CUDA, 9)
        with scope.DeviceScope(dsc):
            self.assertEquals(scope.CurrentDeviceScope(), dsc)

        self.assertEquals(scope.CurrentDeviceScope(), None)

    def testEmptyDevicescopeBasic(self):
        self.assertEquals(scope.CurrentDeviceScope(), None)

        dsc = core.DeviceOption(caffe2_pb2.CUDA, 9)
        with scope.DeviceScope(dsc):
            self.assertEquals(scope.CurrentDeviceScope(), dsc)
            with scope.EmptyDeviceScope():
                self.assertEquals(scope.CurrentDeviceScope(), None)
            self.assertEquals(scope.CurrentDeviceScope(), dsc)
        self.assertEquals(scope.CurrentDeviceScope(), None)

    def testDevicescopeAssertion(self):
        self.assertEquals(scope.CurrentDeviceScope(), None)

        dsc = core.DeviceOption(caffe2_pb2.CUDA, 9)

        try:
            with scope.DeviceScope(dsc):
                self.assertEquals(scope.CurrentDeviceScope(), dsc)
                raise Exception()
        except Exception:
            pass

        self.assertEquals(scope.CurrentDeviceScope(), None)

    def testTags(self):
        self.assertEquals(scope.CurrentTags(), None)

        tags1 = {"key1": "value1"}
        tags2 = {"key2": "value2"}
        tags3 = {"key3": "value3"}

        tags_1_2 = tags1.copy()
        tags_1_2.update(tags2)

        tags_1_3 = tags1.copy()
        tags_1_3.update(tags3)

        tags_1_2_3 = tags_1_2.copy()
        tags_1_2_3.update(tags3)

        with scope.Tags(tags1):
            self.assertEquals(scope.CurrentTags(), tags1)

            with scope.Tags(tags2):
                self.assertEquals(scope.CurrentTags(), tags_1_2)

                with scope.Tags(tags3):
                    self.assertEquals(scope.CurrentTags(), tags_1_2_3)

            with scope.Tags(tags2):
                self.assertEquals(scope.CurrentTags(), tags_1_2)

            self.assertEquals(scope.CurrentTags(), tags1)

            with scope.Tags(tags3):
                self.assertEquals(scope.CurrentTags(), tags_1_3)

            self.assertEquals(scope.CurrentTags(), tags1)
        self.assertEquals(scope.CurrentTags(), None)

    def testTagsErrors(self):
        self.assertEquals(scope.CurrentTags(), None)

        tags1 = {"key1": "value1"}
        tags2 = {"key1": "value2"}
        # Test tags value conflict
        with scope.Tags(tags1):
            self.assertEquals(scope.CurrentTags(), tags1)
            try:
                with scope.Tags(tags2):
                    print("CurrentTags:".format(scope.CurrentTags()))
            except ValueError as err:
                    print("Tags conflict detected as expected: {}".format(err))
            else:
                assert False, "Expecting a Tags conflict failure"

    def testTagsAndDeviceScope(self):

        # test empty DeviceScope and empty Tags()
        test_0_net = core.Net("test0")
        test_0_net.Add(["X", "Y"], "Z")
        self.assertEquals(
            test_0_net.Proto().op[0].device_option.extra_info, []
        )

        # test empty DeviceScope and Tags()
        test_1_net = core.Net("test1")
        tags1 = {"key1": "value1"}
        with scope.Tags(tags1):
            test_1_net.Add(["X", "Y"], "Z")
        self.assertEquals(
            test_1_net.Proto().op[0].device_option.extra_info[0], "key1:value1"
        )

        # test DeviceScope with empty extra_info and Tags()
        dev = core.DeviceOption(0)
        test_2_net = core.Net("test2")
        with scope.DeviceScope(dev):
            with scope.Tags(tags1):
                test_2_net.Add(["X", "Y"], "Z")
        self.assertEquals(
            test_2_net.Proto().op[0].device_option.extra_info[0], "key1:value1"
        )

        # test DeviceScope with extra_info and Tags(), without value conflicts
        dev = core.DeviceOption(0, extra_info=["key2:value2"])
        test_3_net = core.Net("test3")
        with scope.DeviceScope(dev):
            with scope.Tags(tags1):
                test_3_net.Add(["X", "Y"], "Z")
        self.assertEquals(
            test_3_net.Proto().op[0].device_option.extra_info,
            ["key1:value1", "key2:value2"],
        )

        # test DeviceScope with extra_info and Tags(), with value conflicts
        dev = core.DeviceOption(0, extra_info=["key1:value2"])
        test_4_net = core.Net("test4")
        try:
            with scope.DeviceScope(dev):
                with scope.Tags(tags1):
                    test_4_net.Add(["X", "Y"], "Z")
        except ValueError as err:
            print("Tags conflict detected as expected: {}".format(err))
        else:
            assert False, "Expecting a Tags conflict failure"

    def testMultiThreaded(self):
        """
        Test that name/device scope are properly local to the thread
        and don't interfere
        """
        global SUCCESS_COUNT
        self.assertEquals(scope.CurrentNameScope(), "")
        self.assertEquals(scope.CurrentDeviceScope(), None)

        threads = []
        for i in range(4):
            threads.append(threading.Thread(
                target=thread_runner,
                args=(i, self),
            ))
        for t in threads:
            t.start()

        with scope.NameScope("master"):
            self.assertEquals(scope.CurrentDeviceScope(), None)
            self.assertEquals(scope.CurrentNameScope(), "master/")
            for t in threads:
                t.join()

            self.assertEquals(scope.CurrentNameScope(), "master/")
            self.assertEquals(scope.CurrentDeviceScope(), None)

        # Ensure all threads succeeded
        self.assertEquals(SUCCESS_COUNT, 4)
