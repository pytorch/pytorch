"""Tests for the ``sympy.physics.biomechanics._mixin.py`` module."""

import pytest

from sympy.physics.biomechanics._mixin import _NamedMixin


class TestNamedMixin:

    @staticmethod
    def test_subclass():

        class Subclass(_NamedMixin):

            def __init__(self, name):
                self.name = name

        instance = Subclass('name')
        assert instance.name == 'name'

    @pytest.fixture(autouse=True)
    def _named_mixin_fixture(self):

        class Subclass(_NamedMixin):

            def __init__(self, name):
                self.name = name

        self.Subclass = Subclass

    @pytest.mark.parametrize('name', ['a', 'name', 'long_name'])
    def test_valid_name_argument(self, name):
        instance = self.Subclass(name)
        assert instance.name == name

    @pytest.mark.parametrize('invalid_name', [0, 0.0, None, False])
    def test_invalid_name_argument_not_str(self, invalid_name):
        with pytest.raises(TypeError):
            _ = self.Subclass(invalid_name)

    def test_invalid_name_argument_zero_length_str(self):
        with pytest.raises(ValueError):
            _ = self.Subclass('')

    def test_name_attribute_is_immutable(self):
        instance = self.Subclass('name')
        with pytest.raises(AttributeError):
            instance.name = 'new_name'
