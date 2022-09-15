import dataclasses


@dataclasses.dataclass
class DiagnosticOptions:
    """
    Options for diagnostic tool.
    """

    verbose: bool = False
