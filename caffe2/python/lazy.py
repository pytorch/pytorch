## @package workspace
# Module caffe2.python.lazy

_import_lazy_calls = []

def RegisterLazyImport(lazy):
    global _import_lazy_calls
    _import_lazy_calls += [lazy]


def TriggerLazyImport():
    global _import_lazy_calls
    for lazy in _import_lazy_calls:
        lazy()
