from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from importlib import import_module
import pkgutil
import sys
from . import layers


def import_recursive(package):
    """
    Takes a package and imports all modules underneath it
    """
    pkg_dir = package.__path__
    module_location = package.__name__
    for (module_loader, name, ispkg) in pkgutil.iter_modules(pkg_dir):
        module_name = "{}.{}".format(module_location, name)  # Module/package
        module = import_module(module_name)
        if ispkg:
            import_recursive(module)

import_recursive(sys.modules[__name__])

for cls in layers.ModelLayer.__subclasses__():
    layers.register_layer(cls.__name__, cls)
