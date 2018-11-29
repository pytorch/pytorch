from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import scope, core, workspace
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
    dsc = core.DeviceOption(workspace.GpuDeviceType, idx)
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

        dsc = core.DeviceOption(workspace.GpuDeviceType, 9)
        with scope.DeviceScope(dsc):
            self.assertEquals(scope.CurrentDeviceScope(), dsc)

        self.assertEquals(scope.CurrentDeviceScope(), None)

    def testEmptyDevicescopeBasic(self):
        self.assertEquals(scope.CurrentDeviceScope(), None)

        dsc = core.DeviceOption(workspace.GpuDeviceType, 9)
        with scope.DeviceScope(dsc):
            self.assertEquals(scope.CurrentDeviceScope(), dsc)
            with scope.EmptyDeviceScope():
                self.assertEquals(scope.CurrentDeviceScope(), None)
            self.assertEquals(scope.CurrentDeviceScope(), dsc)
        self.assertEquals(scope.CurrentDeviceScope(), None)

    def testDevicescopeAssertion(self):
        self.assertEquals(scope.CurrentDeviceScope(), None)

        dsc = core.DeviceOption(workspace.GpuDeviceType, 9)

        try:
            with scope.DeviceScope(dsc):
                self.assertEquals(scope.CurrentDeviceScope(), dsc)
                raise Exception()
        except Exception:
            pass

        self.assertEquals(scope.CurrentDeviceScope(), None)

    def testTags(self):
        self.assertEquals(scope.CurrentDeviceScope(), None)

        extra_info1 = ["key1:value1"]
        extra_info2 = ["key2:value2"]
        extra_info3 = ["key3:value3"]

        extra_info_1_2 = ["key1:value1", "key2:value2"]
        extra_info_1_2_3 = ["key1:value1", "key2:value2", "key3:value3"]

        with scope.DeviceScope(core.DeviceOption(0, extra_info=extra_info1)):
            self.assertEquals(scope.CurrentDeviceScope().extra_info, extra_info1)

            with scope.DeviceScope(core.DeviceOption(0, extra_info=extra_info2)):
                self.assertEquals(scope.CurrentDeviceScope().extra_info, extra_info_1_2)

                with scope.DeviceScope(core.DeviceOption(0, extra_info=extra_info3)):
                    self.assertEquals(
                        scope.CurrentDeviceScope().extra_info, extra_info_1_2_3
                    )

                self.assertEquals(scope.CurrentDeviceScope().extra_info, extra_info_1_2)
            self.assertEquals(scope.CurrentDeviceScope().extra_info, extra_info1)
        self.assertEquals(scope.CurrentDeviceScope(), None)

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
