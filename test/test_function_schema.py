from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from common_utils import TestCase, run_tests


class TestFunctionSchema(TestCase):
    def test_serialize_and_deserialize(self):
        schemas = torch._C._jit_get_all_schemas()
        # so far we have around 1700 registered schemas
        self.assertGreater(len(schemas), 1000)
        for schema in schemas:
            parsed_schema = torch._C.parse_schema(str(schema))
            self.assertEqual(parsed_schema, schema)


if __name__ == '__main__':
    run_tests()
