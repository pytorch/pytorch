import os
from . import CWrapPlugin
from ...shared import import_module

BASE_PATH = os.path.realpath(os.path.join(__file__, '..', '..', '..', '..'))
TENSORLIB_PATH = os.path.join(BASE_PATH, 'torch', 'lib', 'TensorLib',
                              'common_with_cwrap.py')

tensorlib_common = import_module('torch.lib.TensorLib.common_with_cwrap', TENSORLIB_PATH)


class ArgcountSortPlugin(CWrapPlugin):

    def __init__(self, descending=True):
        self.descending = descending

    def process_declarations(self, declarations):
        for declaration in declarations:
            tensorlib_common.sort_by_number_of_options(declaration,
                                                       self.descending)
        return declarations
