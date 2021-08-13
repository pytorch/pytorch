from torch.testing._internal.common_utils import run_tests

from autograd.test_autograd import TestAutograd
from autograd.test_complex import TestAutogradComplex
from autograd.test_device_type import *  # names are not static, see module's __all__
from autograd.test_forward_mode import TestAutogradForwardMode
from autograd.test_functional import TestAutogradFunctional
from autograd.test_inference_mode import TestAutogradInferenceMode
from autograd.test_multithread import TestMultithreadAutograd


if __name__ == '__main__':
    run_tests()
