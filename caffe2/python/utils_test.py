




from caffe2.python import core, utils, test_util

import numpy as np


class TestUtils(test_util.TestCase):
    def testArgsToDict(self):
        args = [utils.MakeArgument("int1", 3),
                utils.MakeArgument("float1", 4.0),
                utils.MakeArgument("string1", "foo"),
                utils.MakeArgument("intlist1", np.array([3, 4])),
                utils.MakeArgument("floatlist1", np.array([5.0, 6.0])),
                utils.MakeArgument("stringlist1", np.array(["foo", "bar"]))]
        dict_ = utils.ArgsToDict(args)
        expected = {"int1" : 3,
                    "float1" : 4.0,
                    "string1" : b"foo",
                    "intlist1" : [3, 4],
                    "floatlist1" : [5.0, 6.0],
                    "stringlist1" : [b"foo", b"bar"]}
        self.assertEqual(dict_, expected, "dictionary version of arguments "
                         "doesn't match original")

    def testBuildUniqueMutexIter(self):
        init_net = core.Net("init_net")
        net = core.Net("net")
        utils.BuildUniqueMutexIter(init_net, net)

        for op in init_net.Proto().op:
            self.assertEqual(op.device_option.extra_info[0],
                             "device_type_override:cpu")

        for op in net.Proto().op:
            self.assertEqual(op.device_option.extra_info[0],
                             "device_type_override:cpu")
