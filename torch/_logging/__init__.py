# Top level logging module for torch logging
# Design doc: https://docs.google.com/document/d/1ZRfTWKa8eaPq1AxaiHrq4ASTPouzzlPiuquSBEJYwS8/edit#
# Simple setup for onboarding (see above doc for more detail):
# 1. register any top-level log qualified name for your module in torch._logging._registrations (see there for examples)
# 2. register any artifacts (<artifact_name> below) in torch._logging._registrations
#   a. call getArtifactLogger(__name__, <artifact_name>) at your logging site instead of the standard logger to log your artifact
import torch._logging._registrations
from ._internal import (
    _init_logs,
    DEFAULT_LOGGING,
    getArtifactLogger,
    LazyString,
    set_logs,
    warning_once,
)
