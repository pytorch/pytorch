from common_utils import TestCase, run_tests
from torch.utils.tensorboardX import SummaryWriter
import tempfile

class TensorboardTest(TestCase):

    def test_create_SummaryWriter(self):
        """Test tensorboard's basic function"""
        with SummaryWriter(tempfile.gettempdir()):
            pass

if __name__ == '__main__':
    run_tests()
