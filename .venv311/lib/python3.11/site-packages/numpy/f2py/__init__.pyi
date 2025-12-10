from .f2py2e import main as main
from .f2py2e import run_main

__all__ = ["get_include", "run_main"]

def get_include() -> str: ...
