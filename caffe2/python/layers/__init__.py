# Copyright (c) 2016-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

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
    for (_module_loader, name, ispkg) in pkgutil.iter_modules(pkg_dir):
        module_name = "{}.{}".format(module_location, name)  # Module/package
        module = import_module(module_name)
        if ispkg:
            import_recursive(module)


def find_subclasses_recursively(base_cls, sub_cls):
    cur_sub_cls = base_cls.__subclasses__()
    sub_cls.update(cur_sub_cls)
    for cls in cur_sub_cls:
        find_subclasses_recursively(cls, sub_cls)


import_recursive(sys.modules[__name__])

model_layer_subcls = set()
find_subclasses_recursively(layers.ModelLayer, model_layer_subcls)

for cls in list(model_layer_subcls):
    layers.register_layer(cls.__name__, cls)
