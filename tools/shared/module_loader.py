import sys


def import_module(name, path):
    if sys.version_info >= (3, 5):
        import importlib.util
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    else:
        from importlib.machinery import SourceFileLoader
        return SourceFileLoader(name, path).load_module()
