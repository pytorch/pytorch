# Top level logging module for torch logging
# Design doc: https://docs.google.com/document/d/1ZRfTWKa8eaPq1AxaiHrq4ASTPouzzlPiuquSBEJYwS8/edit#
# Simple setup for onboarding (see above doc for more detail):
# 1. @loggable any classes you'd like to register as artifacts can be toggled as logged/not logged, and
#    add them to the loggable_types module
#    Only requirement here is that it has a __str__ method, and then instances of this class can be passed directly
#    to log.debug(<instance here>)
# 2. register the top-level log for your component (also in loggable_types)
#    (See loggable_types module for examples or the above design doc)
from ._internal import _init_logs, set_logs, getArtifactLogger

import torch._logging._registrations
