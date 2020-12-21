




from caffe2.python import convert, workspace
from caffe2.proto import caffe2_pb2, torch_pb2
import unittest
import numpy as np

class TestOperator(unittest.TestCase):
    def setUp(self):
        workspace.ResetWorkspace()

if __name__ == '__main__':
    unittest.main()
