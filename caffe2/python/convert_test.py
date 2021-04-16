




from caffe2.python import workspace
import unittest

class TestOperator(unittest.TestCase):
    def setUp(self):
        workspace.ResetWorkspace()

if __name__ == '__main__':
    unittest.main()
