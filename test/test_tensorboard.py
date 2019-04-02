from common_utils import TestCase, run_tests
from torch.utils.tensorboard import SummaryWriter
import tempfile


class TensorBoardTest(TestCase):

    def test_create_summary_writer(self):
        """Test TensorBoard's basic function"""
        with SummaryWriter(tempfile.gettempdir()):
            pass

if __name__ == '__main__':
    run_tests()
