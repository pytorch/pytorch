"""
sympy.polys.matrices.dfm

Provides the :class:`DFM` class if ``GROUND_TYPES=flint'``. Otherwise, ``DFM``
is a placeholder class that raises NotImplementedError when instantiated.
"""

from sympy.external.gmpy import GROUND_TYPES

if GROUND_TYPES == "flint":  # pragma: no cover
    # When python-flint is installed we will try to use it for dense matrices
    # if the domain is supported by python-flint.
    from ._dfm import DFM

else: # pragma: no cover
    # Other code should be able to import this and it should just present as a
    # version of DFM that does not support any domains.
    class DFM_dummy:
        """
        Placeholder class for DFM when python-flint is not installed.
        """
        def __init__(*args, **kwargs):
            raise NotImplementedError("DFM requires GROUND_TYPES=flint.")

        @classmethod
        def _supports_domain(cls, domain):
            return False

        @classmethod
        def _get_flint_func(cls, domain):
            raise NotImplementedError("DFM requires GROUND_TYPES=flint.")

    # mypy really struggles with this kind of conditional type assignment.
    # Maybe there is a better way to annotate this rather than type: ignore.
    DFM = DFM_dummy # type: ignore
