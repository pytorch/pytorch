"""Mixin classes for sharing functionality between unrelated classes.

This module is named with a leading underscore to signify to users that it's
"private" and only intended for internal use by the biomechanics module.

"""


__all__ = ['_NamedMixin']


class _NamedMixin:
    """Mixin class for adding `name` properties.

    Valid names, as will typically be used by subclasses as a suffix when
    naming automatically-instantiated symbol attributes, must be nonzero length
    strings.

    Attributes
    ==========

    name : str
        The name identifier associated with the instance. Must be a string of
        length at least 1.

    """

    @property
    def name(self) -> str:
        """The name associated with the class instance."""
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        if hasattr(self, '_name'):
            msg = (
                f'Can\'t set attribute `name` to {repr(name)} as it is '
                f'immutable.'
            )
            raise AttributeError(msg)
        if not isinstance(name, str):
            msg = (
                f'Name {repr(name)} passed to `name` was of type '
                f'{type(name)}, must be {str}.'
            )
            raise TypeError(msg)
        if name in {''}:
            msg = (
                f'Name {repr(name)} is invalid, must be a nonzero length '
                f'{type(str)}.'
            )
            raise ValueError(msg)
        self._name = name
