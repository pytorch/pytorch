# Owner(s): ["oncall: export"]

from torch._dynamo.test_case import run_tests, TestCase
from torch.export.dynamic_shapes import _DimHint, _DimHintType, Dim


class TestDimHint(TestCase):
    def test_dimhint_repr(self):
        hint = _DimHint(_DimHintType.DYNAMIC)
        self.assertEqual(repr(hint), "DimHint(DYNAMIC)")

        hint_with_bounds = _DimHint(_DimHintType.AUTO, min=1, max=64)
        self.assertEqual(repr(hint_with_bounds), "DimHint(AUTO, min=1, max=64)")

        non_factory_hint = _DimHint(_DimHintType.STATIC, min=16, _factory=False)
        self.assertEqual(repr(non_factory_hint), "DimHint(STATIC, min=16)")

    def test_dimhint_factory(self):
        factory = _DimHint(_DimHintType.AUTO)
        self.assertTrue(factory._factory)

        result = factory(min=8, max=32)
        self.assertEqual(result.type, _DimHintType.AUTO)
        self.assertEqual(result.min, 8)
        self.assertEqual(result.max, 32)
        self.assertFalse(result._factory)

        with self.assertRaises(TypeError) as cm:
            result(min=1, max=10)
        self.assertIn("object is not callable", str(cm.exception))

        bounded = Dim.DYNAMIC(min=4, max=16)
        self.assertEqual(repr(bounded), "DimHint(DYNAMIC, min=4, max=16)")

        with self.assertRaises(AssertionError):
            factory(min=-1)


if __name__ == "__main__":
    run_tests()
