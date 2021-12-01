# Owner(s): ["oncall: mobile"]

from torch.testing._internal.common_utils import TestCase, run_tests
from tools.codegen.operator_versions.gen_mobile_upgraders import parse_upgrader, write_cpp
from pathlib import Path
import tempfile
import os

pytorch_test_dir = Path(__file__).resolve().parents[1]

class TestLiteScriptModule(TestCase):

    def test_generate_bytecode(self):
        upgrader_bytecode_table_path = pytorch_test_dir / "mobile" / "test_upgrader_bytecode_table_example.yaml"
        upgrader_mobile_cpp_path = pytorch_test_dir / "mobile" / "test_upgrader_bytecode_table_example.h"
        upgrader_dict = parse_upgrader(upgrader_bytecode_table_path)
        with tempfile.TemporaryDirectory() as tmpdirname:
            write_cpp(tmpdirname, upgrader_dict)
            with open(os.path.join(tmpdirname, 'upgrader_mobile.h'), 'r') as file_name:
                actual_output = [line.strip() for line in file_name.readlines() if line]
            with open(str(upgrader_mobile_cpp_path), 'r') as file_name:
                expect_output = [line.strip() for line in file_name.readlines() if line]
            actual_output_filtered = list(filter(lambda token: len(token) != 0, actual_output))
            expect_output_filtered = list(filter(lambda token: len(token) != 0, expect_output))
            self.assertEqual(actual_output_filtered, expect_output_filtered)


if __name__ == '__main__':
    run_tests()
