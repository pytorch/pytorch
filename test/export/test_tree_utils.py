# Owner(s): ["oncall: export"]
from collections import OrderedDict

import torch
from torch._dynamo.test_case import TestCase

from torch.export._tree_utils import is_equivalent, reorder_kwargs
from torch.testing._internal.common_utils import run_tests
from torch.utils._pytree import tree_structure


class TestTreeUtils(TestCase):
    def test_reorder_kwargs(self):
        original_kwargs = {"a": torch.tensor(0), "b": torch.tensor(1)}
        user_kwargs = {"b": torch.tensor(2), "a": torch.tensor(3)}
        orig_spec = tree_structure(((), original_kwargs))

        reordered_kwargs = reorder_kwargs(user_kwargs, orig_spec)

        # Key ordering should be the same
        self.assertEqual(reordered_kwargs.popitem()[0], original_kwargs.popitem()[0]),
        self.assertEqual(reordered_kwargs.popitem()[0], original_kwargs.popitem()[0]),

    def test_equivalence_check(self):
        tree1 = {"a": torch.tensor(0), "b": torch.tensor(1), "c": None}
        tree2 = OrderedDict(a=torch.tensor(0), b=torch.tensor(1), c=None)
        spec1 = tree_structure(tree1)
        spec2 = tree_structure(tree2)

        def dict_ordered_dict_eq(type1, context1, type2, context2):
            if type1 is None or type2 is None:
                return type1 is type2 and context1 == context2

            if issubclass(type1, (dict, OrderedDict)) and issubclass(
                type2, (dict, OrderedDict)
            ):
                return context1 == context2

            return type1 is type2 and context1 == context2

        self.assertTrue(is_equivalent(spec1, spec2, dict_ordered_dict_eq))

        # Wrong ordering should still fail
        tree3 = OrderedDict(b=torch.tensor(1), a=torch.tensor(0))
        spec3 = tree_structure(tree3)
        self.assertFalse(is_equivalent(spec1, spec3, dict_ordered_dict_eq))


if __name__ == "__main__":
    run_tests()
