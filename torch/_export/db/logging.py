# mypy: allow-untyped-defs

from .examples import all_examples
from torch._utils_internal import log_export_usage

ALL_EXAMPLES = all_examples()

def exportdb_error_message(case_name: str):
    # Detect whether case_name is really registered in exportdb.
    if case_name in ALL_EXAMPLES:
        url_case_name = case_name.replace("_", "-")
        return f"See {case_name} in exportdb for unsupported case. \
                https://pytorch.org/docs/main/generated/exportdb/index.html#{url_case_name}"
    else:
        log_export_usage(
            event="export.error.casenotregistered",
            message=case_name,
        )
        return f"{case_name} is unsupported."


def get_class_if_classified_error(e):
    from torch._dynamo.exc import TorchRuntimeError, Unsupported, UserError

    _ALLOW_LIST = {
        Unsupported,
        UserError,
        TorchRuntimeError,
    }
    case_name = getattr(e, "case_name", None)
    if type(e) in _ALLOW_LIST and case_name is not None:
        return case_name
    return None
