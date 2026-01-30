# Owner(s): ["module: PrivateUse1"]
from torch.testing._internal.common_utils import run_tests, TestCase


class TestDeviceBackendLogging(TestCase):
    def test_log(self):
        from torch_mock_backend.logger_registration import test_logger

        test_logger()


if __name__ == "__main__":
    run_tests()
