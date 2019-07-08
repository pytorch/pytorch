from . import CWrapPlugin
from ...shared import cwrap_common


class OptionalArguments(CWrapPlugin):

    def process_declarations(self, declarations):
        for declaration in declarations:
            cwrap_common.enumerate_options_due_to_default(
                declaration,
                allow_kwarg=True,
                type_to_signature={},
                remove_self=False)

        return declarations
