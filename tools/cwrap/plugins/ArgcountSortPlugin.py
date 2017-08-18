import os
from . import CWrapPlugin
from ...shared import cwrap_common


class ArgcountSortPlugin(CWrapPlugin):

    def __init__(self, descending=True):
        self.descending = descending

    def process_declarations(self, declarations):
        for declaration in declarations:
            cwrap_common.sort_by_number_of_options(declaration,
                                                   self.descending)
        return declarations
