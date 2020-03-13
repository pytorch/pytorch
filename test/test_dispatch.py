# import torch
import torch._C as C
from torch.testing._internal.common_utils import TestCase, run_tests

class TestDispatch(TestCase):
    def test_def(self):
        m = C._dispatch_import()
        m.def_("arfy(Tensor x) -> Tensor")
        m.impl_t_t("arfy")
        print(C._dispatch_dump("arfy"))

if __name__ == '__main__':
    run_tests()
