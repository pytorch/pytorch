from ._C import *

from .nvfuser_version import __version__

def version():
    r"""returns nvfuser version in format of a string 'm.n.p+git[7d-sha]'.

    We strip the git[7d-sha] and convert the string to
    `packaging.version.Version` for comparison. e.g. you can use it as:
        import nvfuser
        print(nvfuser.version())              # 0.0.1+git21df524
        nvfuser.version() == '0.0.1`          # True
        nvfuser.version() > '0.0.0`           # True

        from packaging.version import Version
        nvfuser.version() < Version('1.0.0')  # True
    """
    return __version__
