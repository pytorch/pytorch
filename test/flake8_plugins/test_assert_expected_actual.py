import ast
from flake8_plugins.assert_expected_actual import Plugin


def lints(source_code):
    plugin = Plugin(ast.parse(source_code), 'foo.py')
    return {f'{line}:{col+1} {msg}' for line, col, msg, _ in plugin.run()}


def test_empty_program():
    assert set() == lints('')


def test_correct_order():
    assert set() == lints('self.assertEqual(expected, actual)')


def test_two_literals():
    assert set() == lints('self.assertEqual(3, [4, 5])')


def test_left_literal():
    assert set() == lints('self.assertEqual(42, foo)')


def test_neither_contains_string():
    assert set() == lints('self.assertEqual(foo, bar)')
    assert set() == lints('self.assertEqual(bar, foo)')


def test_incorrect_order():
    ret = lints('self.assertEqual(actual, expected)')
    assert {'1:1 PTA100 expected should come before actual'} == ret


def test_only_actual():
    ret = lints('self.assertEqual(actual, foo)')
    assert {'1:1 PTA100 expected should come before actual'} == ret


def test_only_expected():
    ret = lints('self.assertEqual(foo, expected)')
    assert {'1:1 PTA100 expected should come before actual'} == ret


def test_ignore_case_actual():
    ret = lints('self.assertEqual(aCtUaL, 42)')
    assert {'1:1 PTA100 expected should come before actual'} == ret


def test_ignore_case_expected():
    ret = lints('self.assertEqual(baz, ExPeCtEd)')
    assert {'1:1 PTA100 expected should come before actual'} == ret


def test_right_literal():
    ret = lints('self.assertEqual(foo, 42)')
    assert {'1:1 PTA100 expected should come before actual'} == ret


def test_right_complicated_literal():
    ret = lints('self.assertEqual(foo, [42, 34, "cat"])')
    assert {'1:1 PTA100 expected should come before actual'} == ret
