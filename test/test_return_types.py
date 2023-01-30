# Owner(s): ["module: autograd"]

import inspect

import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.common_utils import instantiate_parametrized_tests


def is_namedtuple_class(cls):
    bases = cls.__bases__
    if len(bases) != 1 or bases[0] != tuple:
        return False
    fields = getattr(cls, '_fields', None)
    if not isinstance(fields, tuple):
        return False
    return all(type(entry) is str for entry in fields)


class TestReturnTypes(TestCase):
    def test_hasattr_fields(self):
        for name in torch.return_types.__all__:
            attr = getattr(torch.return_types, name)
            if inspect.isclass(attr) and issubclass(attr, tuple):
                self.assertTrue(is_namedtuple_class(attr))


instantiate_parametrized_tests(TestReturnTypes)

if __name__ == '__main__':
    run_tests()
