from pybind11_tests import docstring_options as m


def test_docstring_options():
    # options.disable_function_signatures()
    assert not m.test_function1.__doc__

    assert m.test_function2.__doc__ == "A custom docstring"

    # docstring specified on just the first overload definition:
    assert m.test_overloaded1.__doc__ == "Overload docstring"

    # docstring on both overloads:
    assert m.test_overloaded2.__doc__ == "overload docstring 1\noverload docstring 2"

    # docstring on only second overload:
    assert m.test_overloaded3.__doc__ == "Overload docstr"

    # options.enable_function_signatures()
    assert m.test_function3.__doc__ .startswith("test_function3(a: int, b: int) -> None")

    assert m.test_function4.__doc__ .startswith("test_function4(a: int, b: int) -> None")
    assert m.test_function4.__doc__ .endswith("A custom docstring\n")

    # options.disable_function_signatures()
    # options.disable_user_defined_docstrings()
    assert not m.test_function5.__doc__

    # nested options.enable_user_defined_docstrings()
    assert m.test_function6.__doc__ == "A custom docstring"

    # RAII destructor
    assert m.test_function7.__doc__ .startswith("test_function7(a: int, b: int) -> None")
    assert m.test_function7.__doc__ .endswith("A custom docstring\n")

    # Suppression of user-defined docstrings for non-function objects
    assert not m.DocstringTestFoo.__doc__
    assert not m.DocstringTestFoo.value_prop.__doc__
