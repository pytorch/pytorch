# mypy: allow-untyped-defs


def exportdb_error_message(case_name: str):
    from .examples import all_examples
    from torch._utils_internal import log_export_usage

    ALL_EXAMPLES = all_examples()
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
    """
    Returns a string case name if the export error e is classified.
    Returns None otherwise.
    """

    from torch._dynamo.exc import TorchRuntimeError, Unsupported, UserError

    ALWAYS_CLASSIFIED = "always_classified"
    DEFAULT_CLASS_SIGIL = "case_name"

    # add error types that should be classified, along with any attribute name
    # whose presence acts like a sigil to further distinguish which errors of
    # that type should be classified. If the attribute name is None, then the
    # error type is always classified.
    _ALLOW_LIST = {
        Unsupported: DEFAULT_CLASS_SIGIL,
        UserError: DEFAULT_CLASS_SIGIL,
        TorchRuntimeError: None,
    }
    if type(e) in _ALLOW_LIST:
        attr_name = _ALLOW_LIST[type(e)]
        if attr_name is None:
            return ALWAYS_CLASSIFIED
        return getattr(e, attr_name, None)
    return None
