from sympy.external import import_module
from sympy.testing.pytest import warns

# fixes issue that arose in addressing issue 6533
def test_no_stdlib_collections():
    '''
    make sure we get the right collections when it is not part of a
    larger list
    '''
    import collections
    matplotlib = import_module('matplotlib',
        import_kwargs={'fromlist': ['cm', 'collections']},
        min_module_version='1.1.0', catch=(RuntimeError,))
    if matplotlib:
        assert collections != matplotlib.collections

def test_no_stdlib_collections2():
    '''
    make sure we get the right collections when it is not part of a
    larger list
    '''
    import collections
    matplotlib = import_module('matplotlib',
        import_kwargs={'fromlist': ['collections']},
        min_module_version='1.1.0', catch=(RuntimeError,))
    if matplotlib:
        assert collections != matplotlib.collections

def test_no_stdlib_collections3():
    '''make sure we get the right collections with no catch'''
    import collections
    matplotlib = import_module('matplotlib',
        import_kwargs={'fromlist': ['cm', 'collections']},
        min_module_version='1.1.0')
    if matplotlib:
        assert collections != matplotlib.collections

def test_min_module_version_python3_basestring_error():
    with warns(UserWarning):
        import_module('mpmath', min_module_version='1000.0.1')
