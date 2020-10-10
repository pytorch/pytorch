"""
Module defining global singleton classes.

This module raises a RuntimeError if an attempt to reload it is made. In that
way the identities of the classes defined here are fixed and will remain so
even if numpy itself is reloaded. In particular, a function like the following
will still work correctly after numpy is reloaded::

    def foo(arg=np._NoValue):
        if arg is np._NoValue:
            ...

That was not the case when the singleton classes were defined in the numpy
``__init__.py`` file. See gh-7844 for a discussion of the reload problem that
motivated this module.

"""
__ALL__ = [
    'ModuleDeprecationWarning', 'VisibleDeprecationWarning', '_NoValue'
    ]


# Disallow reloading this module so as to preserve the identities of the
# classes defined here.
if '_is_loaded' in globals():
    raise RuntimeError('Reloading numpy._globals is not allowed')
_is_loaded = True


class ModuleDeprecationWarning(DeprecationWarning):
    """Module deprecation warning.

    The nose tester turns ordinary Deprecation warnings into test failures.
    That makes it hard to deprecate whole modules, because they get
    imported by default. So this is a special Deprecation warning that the
    nose tester will let pass without making tests fail.

    """


ModuleDeprecationWarning.__module__ = 'numpy'


class VisibleDeprecationWarning(UserWarning):
    """Visible deprecation warning.

    By default, python will not show deprecation warnings, so this class
    can be used when a very visible warning is helpful, for example because
    the usage is most likely a user bug.

    """


VisibleDeprecationWarning.__module__ = 'numpy'


class _NoValueType:
    """Special keyword value.

    The instance of this class may be used as the default value assigned to a
    deprecated keyword in order to check if it has been given a user defined
    value.
    """
    __instance = None
    def __new__(cls):
        # ensure that only one instance exists
        if not cls.__instance:
            cls.__instance = super(_NoValueType, cls).__new__(cls)
        return cls.__instance

    # needed for python 2 to preserve identity through a pickle
    def __reduce__(self):
        return (self.__class__, ())

    def __repr__(self):
        return "<no value>"


_NoValue = _NoValueType()
