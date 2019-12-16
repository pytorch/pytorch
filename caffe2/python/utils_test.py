from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from caffe2.python import core, test_util, utils


class TestUtils(test_util.TestCase):
    def testArgsToDict(self):
        args = [
            utils.MakeArgument("int1", 3),
            utils.MakeArgument("float1", 4.0),
            utils.MakeArgument("string1", "foo"),
            utils.MakeArgument("intlist1", np.array([3, 4])),
            utils.MakeArgument("floatlist1", np.array([5.0, 6.0])),
            utils.MakeArgument("stringlist1", np.array(["foo", "bar"])),
        ]
        dict_ = utils.ArgsToDict(args)
        expected = {
            "int1": 3,
            "float1": 4.0,
            "string1": b"foo",
            "intlist1": [3, 4],
            "floatlist1": [5.0, 6.0],
            "stringlist1": [b"foo", b"bar"],
        }
        self.assertEqual(
            dict_, expected, "dictionary version of arguments " "doesn't match original"
        )

    def testOpEqualExceptDebugInfo(self):
        def create_two_ops(op_name_1, op_name_2):
            input = core.BlobReference("input_blob_name")
            output = core.BlobReference("output_blob_name")
            return (
                core.CreateOperator("Cast", input, output, name=op_name_1),
                core.CreateOperator(
                    "Cast", input, output, name=op_name_2, debug_info="second op"
                ),
            )

        op1, op2 = create_two_ops("op_name", "op_name")
        self.assertTrue(utils.OpEqualExceptDebugInfo(op1, op2))
        # debug info doesn't change after OpEqualExceptDebugInfo. op1 has
        # empty debug info by default.
        self.assertEqual(op1.debug_info, "")
        self.assertEqual(op2.debug_info, "second op")

        op3, op4 = create_two_ops("name_1", "name_2")
        self.assertFalse(utils.OpEqualExceptDebugInfo(op3, op4))
