# Owner(s): ["oncall: mobile"]

from torch.testing._internal.common_utils import TestCase, run_tests

from torchgen.operator_versions.gen_mobile_upgraders import (
    sort_upgrader,
    write_cpp,
)
from pathlib import Path
import tempfile
import os
from torch.jit.generate_bytecode import generate_upgraders_bytecode

pytorch_caffe2_dir = Path(__file__).resolve().parents[2]

class TestLiteScriptModule(TestCase):

    def test_generate_bytecode(self):
        upgrader_list = generate_upgraders_bytecode()
        sorted_upgrader_list = sort_upgrader(upgrader_list)
        upgrader_mobile_cpp_path = pytorch_caffe2_dir / "torch" / "csrc" / "jit" / "mobile" / "upgrader_mobile.cpp"
        with tempfile.TemporaryDirectory() as tmpdirname:
            write_cpp(tmpdirname, sorted_upgrader_list)
            with open(os.path.join(tmpdirname, 'upgrader_mobile.cpp')) as file_name:
                actual_output = [line.strip() for line in file_name.readlines() if line]
            with open(str(upgrader_mobile_cpp_path)) as file_name:
                expect_output = [line.strip() for line in file_name.readlines() if line]
            actual_output_filtered = list(filter(lambda token: len(token) != 0, actual_output))
            expect_output_filtered = list(filter(lambda token: len(token) != 0, expect_output))

            self.assertEqual(actual_output_filtered, expect_output_filtered)

if __name__ == '__main__':
    run_tests()
