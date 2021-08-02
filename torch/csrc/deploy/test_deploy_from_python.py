from libfb.py import testutil
import sys
import os

import test_deploy_python_ext

class TestDeployFromPython(testutil.BaseFacebookTestCase):
    def test_deploy_from_python(self):
        self.assertTrue(test_deploy_python_ext.run())
