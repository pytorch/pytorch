from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from importlib import import_module
import pkgutil
import sys
import inspect
from . import layers


def import_recursive(package, clsmembers):
    """
    Takes a package and imports all modules underneath it
    """

    pkg_dir = package.__path__
    module_location = package.__name__
    for (_module_loader, name, ispkg) in pkgutil.iter_modules(pkg_dir):
        module_name = "{}.{}".format(module_location, name)  # Module/package
        module = import_module(module_name)
        clsmembers += [cls[1] for cls in inspect.getmembers(module, inspect.isclass)]
        if ispkg:
            import_recursive(module, clsmembers)


clsmembers = []
import_recursive(sys.modules[__name__], clsmembers)

for cls in clsmembers:
    if issubclass(cls, layers.ModelLayer) and cls is not layers.ModelLayer:
        layers.register_layer(cls.__name__, cls)
