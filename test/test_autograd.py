from torch.testing._internal.common_utils import run_tests

# F401 is unused identifiers. we suppress this because these
# identifiers are discovered and used by virtue of their presence in
# the global scope by run_tests().

from autograd.test_autograd import TestAutograd  # noqa: F401
from autograd.test_complex import TestAutogradComplex  # noqa: F401
# The device specific tests can vary synchronically with different
# compilation options (e.g. with or without CUDA) and diachronically
# as we add or remove devices. We don't know what tests are generated
# in this module and thus import what it decides to export (see
# __all__ in the module).
from autograd.test_device_type import *  # noqa: F401, F403
from autograd.test_forward_mode import TestAutogradForwardMode  # noqa: F401
from autograd.test_functional import TestAutogradFunctional  # noqa: F401
from autograd.test_inference_mode import TestAutogradInferenceMode  # noqa: F401
from autograd.test_multithread import TestMultithreadAutograd  # noqa: F401


if __name__ == '__main__':
    run_tests()
