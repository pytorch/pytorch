"""Adapted from https://github.com/arogozhnikov/einops/blob/master/tests/test_parsing.py."""
from functorch.dim.einops._parsing import AnonymousAxis, ParsedExpression, _ellipsis
from torch.testing._internal.common_utils import TestCase, run_tests
from typing import Callable
from unittest import mock

mock_anonymous_axis_eq: Callable[[AnonymousAxis, object], bool] = (
    lambda self, other: isinstance(other, AnonymousAxis) and self.value == other.value
)


class TestAnonymousAxis(TestCase):
    def test_anonymous_axes(self) -> None:
        a, b = AnonymousAxis('2'), AnonymousAxis('2')
        self.assertNotEqual(a, b)

        with mock.patch.object(AnonymousAxis, '__eq__', mock_anonymous_axis_eq):
            c, d = AnonymousAxis('2'), AnonymousAxis('3')
            self.assertEqual(a, c)
            self.assertEqual(b, c)
            self.assertNotEqual(a, d)
            self.assertNotEqual(b, d)
            self.assertListEqual([a, 2, b], [c, 2, c])


class TestParsedExpression(TestCase):
    def test_elementary_axis_name(self) -> None:
        for name in ['a', 'b', 'h', 'dx', 'h1', 'zz', 'i9123', 'somelongname',
                     'Alex', 'camelCase', 'u_n_d_e_r_score', 'unreasonablyLongAxisName']:
            self.assertTrue(ParsedExpression.check_axis_name(name))

        for name in ['', '2b', '12', '_startWithUnderscore', 'endWithUnderscore_', '_', '...', _ellipsis]:
            self.assertFalse(ParsedExpression.check_axis_name(name))

    def test_invalid_expressions(self) -> None:
        # double ellipsis should raise an error
        ParsedExpression('... a b c d')
        with self.assertRaises(ValueError):
            ParsedExpression('... a b c d ...')
        with self.assertRaises(ValueError):
            ParsedExpression('... a b c (d ...)')
        with self.assertRaises(ValueError):
            ParsedExpression('(... a) b c (d ...)')

        # double/missing/enclosed parenthesis
        ParsedExpression('(a) b c (d ...)')
        with self.assertRaises(ValueError):
            ParsedExpression('(a)) b c (d ...)')
        with self.assertRaises(ValueError):
            ParsedExpression('(a b c (d ...)')
        with self.assertRaises(ValueError):
            ParsedExpression('(a) (()) b c (d ...)')
        with self.assertRaises(ValueError):
            ParsedExpression('(a) ((b c) (d ...))')

        # invalid identifiers
        ParsedExpression('camelCase under_scored cApiTaLs ß ...')
        with self.assertRaises(ValueError):
            ParsedExpression('1a')
        with self.assertRaises(ValueError):
            ParsedExpression('_pre')
        with self.assertRaises(ValueError):
            ParsedExpression('...pre')
        with self.assertRaises(ValueError):
            ParsedExpression('pre...')

    @mock.patch.object(AnonymousAxis, '__eq__', mock_anonymous_axis_eq)
    def test_parse_expression(self, *mocks: mock.MagicMock) -> None:
        parsed = ParsedExpression('a1  b1   c1    d1')
        self.assertSetEqual(parsed.identifiers, {'a1', 'b1', 'c1', 'd1'})
        self.assertListEqual(parsed.composition, [['a1'], ['b1'], ['c1'], ['d1']])
        self.assertFalse(parsed.has_non_unitary_anonymous_axes)
        self.assertFalse(parsed.has_ellipsis)

        parsed = ParsedExpression('() () () ()')
        self.assertSetEqual(parsed.identifiers, set())
        self.assertListEqual(parsed.composition, [[], [], [], []])
        self.assertFalse(parsed.has_non_unitary_anonymous_axes)
        self.assertFalse(parsed.has_ellipsis)

        parsed = ParsedExpression('1 1 1 ()')
        self.assertSetEqual(parsed.identifiers, set())
        self.assertListEqual(parsed.composition, [[], [], [], []])
        self.assertFalse(parsed.has_non_unitary_anonymous_axes)
        self.assertFalse(parsed.has_ellipsis)

        parsed = ParsedExpression('5 (3 4)')
        self.assertEqual(len(parsed.identifiers), 3)
        self.assertSetEqual({i.value if isinstance(i, AnonymousAxis) else i for i in parsed.identifiers}, {3, 4, 5})
        self.assertListEqual(parsed.composition, [[AnonymousAxis('5')], [AnonymousAxis('3'), AnonymousAxis('4')]])
        self.assertTrue(parsed.has_non_unitary_anonymous_axes)
        self.assertFalse(parsed.has_ellipsis)

        parsed = ParsedExpression('5 1 (1 4) 1')
        self.assertEqual(len(parsed.identifiers), 2)
        self.assertSetEqual({i.value if isinstance(i, AnonymousAxis) else i for i in parsed.identifiers}, {4, 5})
        self.assertListEqual(parsed.composition, [[AnonymousAxis('5')], [], [AnonymousAxis('4')], []])

        parsed = ParsedExpression('name1 ... a1 12 (name2 14)')
        self.assertEqual(len(parsed.identifiers), 6)
        self.assertEqual(len(parsed.identifiers - {'name1', _ellipsis, 'a1', 'name2'}), 2)
        self.assertListEqual(
            parsed.composition, [['name1'], _ellipsis, ['a1'], [AnonymousAxis('12')], ['name2', AnonymousAxis('14')]]
        )
        self.assertTrue(parsed.has_non_unitary_anonymous_axes)
        self.assertTrue(parsed.has_ellipsis)
        self.assertFalse(parsed.has_ellipsis_parenthesized)

        parsed = ParsedExpression('(name1 ... a1 12) name2 14')
        self.assertEqual(len(parsed.identifiers), 6)
        self.assertEqual(len(parsed.identifiers - {'name1', _ellipsis, 'a1', 'name2'}), 2)
        self.assertListEqual(
            parsed.composition, [['name1', _ellipsis, 'a1', AnonymousAxis('12')], ['name2'], [AnonymousAxis('14')]]
        )
        self.assertTrue(parsed.has_non_unitary_anonymous_axes)
        self.assertTrue(parsed.has_ellipsis)
        self.assertTrue(parsed.has_ellipsis_parenthesized)


if __name__ == '__main__':
    run_tests()
