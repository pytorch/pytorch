from libfb.py import testutil
import sys, os

# in order to make sure torch_deploy has all the torch symbols loaded globally,
# make sure it gets opened with RTLD_GLOBAL
sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_LAZY)
import test_deploy_python_ext


class TestDeployFromPython(testutil.BaseFacebookTestCase):
    def test_deploy_from_python(self):
        self.assertTrue(test_deploy_python_ext.run())
