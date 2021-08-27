
import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch._C import parse_schema


class TestForwardCompatibility(TestCase):
    def test_forward_compatibility_for_all_schemas(self):
        schemas = torch._C._jit_get_all_schemas()
        schemas += torch._C._jit_get_custom_class_schemas()

        # load old schemas
        old_schemas = {}
        with open("old_schemas.txt", 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parsed_schema = parse_schema(line)
                old_schemas[parsed_schema.name] = parsed_schema

        for schema in schemas:
            if schema.name not in old_schemas:
                self.assertFalse(True)
            print(schema)
            print(old_schemas[schema.name])
            self.assertTrue(schema.is_forward_compatible_with(old_schemas[schema.name]))

if __name__ == '__main__':
    run_tests()
