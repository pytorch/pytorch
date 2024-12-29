import warnings

from sympy.testing.pytest import (raises, warns, ignore_warnings,
                                    warns_deprecated_sympy, Failed)
from sympy.utilities.exceptions import sympy_deprecation_warning



# Test callables


def test_expected_exception_is_silent_callable():
    def f():
        raise ValueError()
    raises(ValueError, f)


# Under pytest raises will raise Failed rather than AssertionError
def test_lack_of_exception_triggers_AssertionError_callable():
    try:
        raises(Exception, lambda: 1 + 1)
        assert False
    except Failed as e:
        assert "DID NOT RAISE" in str(e)


def test_unexpected_exception_is_passed_through_callable():
    def f():
        raise ValueError("some error message")
    try:
        raises(TypeError, f)
        assert False
    except ValueError as e:
        assert str(e) == "some error message"

# Test with statement

def test_expected_exception_is_silent_with():
    with raises(ValueError):
        raise ValueError()


def test_lack_of_exception_triggers_AssertionError_with():
    try:
        with raises(Exception):
            1 + 1
        assert False
    except Failed as e:
        assert "DID NOT RAISE" in str(e)


def test_unexpected_exception_is_passed_through_with():
    try:
        with raises(TypeError):
            raise ValueError("some error message")
        assert False
    except ValueError as e:
        assert str(e) == "some error message"

# Now we can use raises() instead of try/catch
# to test that a specific exception class is raised


def test_second_argument_should_be_callable_or_string():
    raises(TypeError, lambda: raises("irrelevant", 42))


def test_warns_catches_warning():
    with warnings.catch_warnings(record=True) as w:
        with warns(UserWarning):
            warnings.warn('this is the warning message')
        assert len(w) == 0


def test_warns_raises_without_warning():
    with raises(Failed):
        with warns(UserWarning):
            pass


def test_warns_hides_other_warnings():
    with raises(RuntimeWarning):
        with warns(UserWarning):
            warnings.warn('this is the warning message', UserWarning)
            warnings.warn('this is the other message', RuntimeWarning)


def test_warns_continues_after_warning():
    with warnings.catch_warnings(record=True) as w:
        finished = False
        with warns(UserWarning):
            warnings.warn('this is the warning message')
            finished = True
        assert finished
        assert len(w) == 0


def test_warns_many_warnings():
    with warns(UserWarning):
        warnings.warn('this is the warning message', UserWarning)
        warnings.warn('this is the other warning message', UserWarning)


def test_warns_match_matching():
    with warnings.catch_warnings(record=True) as w:
        with warns(UserWarning, match='this is the warning message'):
            warnings.warn('this is the warning message', UserWarning)
        assert len(w) == 0


def test_warns_match_non_matching():
    with warnings.catch_warnings(record=True) as w:
        with raises(Failed):
            with warns(UserWarning, match='this is the warning message'):
                warnings.warn('this is not the expected warning message', UserWarning)
        assert len(w) == 0

def _warn_sympy_deprecation(stacklevel=3):
    sympy_deprecation_warning(
        "feature",
        active_deprecations_target="active-deprecations",
        deprecated_since_version="0.0.0",
        stacklevel=stacklevel,
    )

def test_warns_deprecated_sympy_catches_warning():
    with warnings.catch_warnings(record=True) as w:
        with warns_deprecated_sympy():
            _warn_sympy_deprecation()
        assert len(w) == 0


def test_warns_deprecated_sympy_raises_without_warning():
    with raises(Failed):
        with warns_deprecated_sympy():
            pass

def test_warns_deprecated_sympy_wrong_stacklevel():
    with raises(Failed):
        with warns_deprecated_sympy():
            _warn_sympy_deprecation(stacklevel=1)

def test_warns_deprecated_sympy_doesnt_hide_other_warnings():
    # Unlike pytest's deprecated_call, we should not hide other warnings.
    with raises(RuntimeWarning):
        with warns_deprecated_sympy():
            _warn_sympy_deprecation()
            warnings.warn('this is the other message', RuntimeWarning)


def test_warns_deprecated_sympy_continues_after_warning():
    with warnings.catch_warnings(record=True) as w:
        finished = False
        with warns_deprecated_sympy():
            _warn_sympy_deprecation()
            finished = True
        assert finished
        assert len(w) == 0

def test_ignore_ignores_warning():
    with warnings.catch_warnings(record=True) as w:
        with ignore_warnings(UserWarning):
            warnings.warn('this is the warning message')
        assert len(w) == 0


def test_ignore_does_not_raise_without_warning():
    with warnings.catch_warnings(record=True) as w:
        with ignore_warnings(UserWarning):
            pass
        assert len(w) == 0


def test_ignore_allows_other_warnings():
    with warnings.catch_warnings(record=True) as w:
        # This is needed when pytest is run as -Werror
        # the setting is reverted at the end of the catch_Warnings block.
        warnings.simplefilter("always")
        with ignore_warnings(UserWarning):
            warnings.warn('this is the warning message', UserWarning)
            warnings.warn('this is the other message', RuntimeWarning)
        assert len(w) == 1
        assert isinstance(w[0].message, RuntimeWarning)
        assert str(w[0].message) == 'this is the other message'


def test_ignore_continues_after_warning():
    with warnings.catch_warnings(record=True) as w:
        finished = False
        with ignore_warnings(UserWarning):
            warnings.warn('this is the warning message')
            finished = True
        assert finished
        assert len(w) == 0


def test_ignore_many_warnings():
    with warnings.catch_warnings(record=True) as w:
        # This is needed when pytest is run as -Werror
        # the setting is reverted at the end of the catch_Warnings block.
        warnings.simplefilter("always")
        with ignore_warnings(UserWarning):
            warnings.warn('this is the warning message', UserWarning)
            warnings.warn('this is the other message', RuntimeWarning)
            warnings.warn('this is the warning message', UserWarning)
            warnings.warn('this is the other message', RuntimeWarning)
            warnings.warn('this is the other message', RuntimeWarning)
        assert len(w) == 3
        for wi in w:
            assert isinstance(wi.message, RuntimeWarning)
            assert str(wi.message) == 'this is the other message'
