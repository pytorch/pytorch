from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_utils import TestCase, run_tests
from functorch_lagging_op_db import (
    functorch_lagging_op_db,
    in_functorch_lagging_op_db,
)
import torch


class TestFuncTorchLaggingOpDb(TestCase):
    def test_functorch_lagging_op_db_has_opinfos(self, device):
        self.assertEqual(len(functorch_lagging_op_db), len(op_db))

    @ops(op_db, allowed_dtypes=(torch.float,))
    def test_coverage(self, device, dtype, op):
        if in_functorch_lagging_op_db(op):
            return
        raise RuntimeError(
            f"{(op.name, op.variant_test_name)} is in PyTorch's OpInfo db ",
            "but is not in functorch's OpInfo db. Please regenerate ",
            "test/functorch_lagging_op_db.py and add the new tests to ",
            "denylists if necessary.")


instantiate_device_type_tests(
    TestFuncTorchLaggingOpDb, globals(), only_for=['cpu'])

if __name__ == '__main__':
    run_tests()
