from typing_extensions import deprecated, override

from ._backend import Backend

class DistutilsBackend(Backend):
    @deprecated(
        "distutils has been deprecated since NumPy 1.26.x. Use the Meson backend instead, or generate wrappers without -c and "
        "use a custom build script"
    )
    # NOTE: the `sef` typo matches runtime
    def __init__(sef, *args: object, **kwargs: object) -> None: ...
    @override
    def compile(self) -> None: ...
