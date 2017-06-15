import os
from copy import deepcopy
from . import CWrapPlugin
from itertools import product
from ...shared import import_module

BASE_PATH = os.path.realpath(os.path.join(__file__, '..', '..', '..', '..'))
TENSORLIB_PATH = os.path.join(BASE_PATH, 'torch', 'lib', 'TensorLib',
                              'common_with_cwrap.py')

tensorlib_common = import_module('torch.lib.TensorLib.common_with_cwrap', TENSORLIB_PATH)


class OptionalArguments(CWrapPlugin):

    def process_declarations(self, declarations):
        for declaration in declarations:
            tensorlib_common.enumerate_options_due_to_default(
                declaration,
                allow_kwarg=True,
                type_to_signature={},
                remove_self=False)

        return declarations
