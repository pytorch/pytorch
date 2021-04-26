from torch.jit.mobile import _get_runtime_bytecode_version
from torch.testing._internal.common_utils import TestCase, run_tests

class testVariousModelVersions(TestCase):
    def test_get_runtime_bytecode_version(self):
        runtime_bytecode_version = _get_runtime_bytecode_version()
        assert(isinstance(runtime_bytecode_version, int))

if __name__ == '__main__':
    run_tests()
