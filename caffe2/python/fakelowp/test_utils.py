from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import numpy as np


def print_test_debug_info(testname, items_dict):
    filename = "debug_operator_onnxifi_" + testname + ".txt"
    np.set_printoptions(threshold=sys.maxsize)
    with open(filename, 'w') as f:
        for key, value in items_dict.items():
            print(key, value)
            f.write("{}\n".format(key))
            f.write("{}\n".format(value))


def print_net(net):
    for i in net.external_input:
        print("Input: {}".format(i))
    for i in net.external_output:
        print("Output: {}".format(i))
    for op in net.op:
        print("Op {}".format(op.type))
        for x in op.input:
            print("  input: {}".format(x))
        for y in op.output:
            print("  output: {}".format(y))
