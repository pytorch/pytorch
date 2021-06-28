from torch.testing._internal.common_utils import TestCase, run_tests

class TestDummy(TestCase):
    def test_dummy_should_fail(self):
        self.assertEqual(5,6)

if __name__ == '__main__':
    run_tests()
