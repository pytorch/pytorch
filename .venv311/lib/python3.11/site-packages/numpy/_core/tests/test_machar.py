"""
Test machar. Given recent changes to hardcode type data, we might want to get
rid of both MachAr and this test at some point.

"""
import numpy._core.numerictypes as ntypes
from numpy import array, errstate
from numpy._core._machar import MachAr


class TestMachAr:
    def _run_machar_highprec(self):
        # Instantiate MachAr instance with high enough precision to cause
        # underflow
        try:
            hiprec = ntypes.float96
            MachAr(lambda v: array(v, hiprec))
        except AttributeError:
            # Fixme, this needs to raise a 'skip' exception.
            "Skipping test: no ntypes.float96 available on this platform."

    def test_underlow(self):
        # Regression test for #759:
        # instantiating MachAr for dtype = np.float96 raises spurious warning.
        with errstate(all='raise'):
            try:
                self._run_machar_highprec()
            except FloatingPointError as e:
                msg = f"Caught {e} exception, should not have been raised."
                raise AssertionError(msg)
