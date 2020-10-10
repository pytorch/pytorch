"""See https://github.com/numpy/numpy/pull/10676.

"""
import sys
import pytest

from numpy.testing import assert_equal
from . import util


class TestQuotedCharacter(util.F2PyTest):
    code = """
      SUBROUTINE FOO(OUT1, OUT2, OUT3, OUT4, OUT5, OUT6)
      CHARACTER SINGLE, DOUBLE, SEMICOL, EXCLA, OPENPAR, CLOSEPAR
      PARAMETER (SINGLE="'", DOUBLE='"', SEMICOL=';', EXCLA="!",
     1           OPENPAR="(", CLOSEPAR=")")
      CHARACTER OUT1, OUT2, OUT3, OUT4, OUT5, OUT6
Cf2py intent(out) OUT1, OUT2, OUT3, OUT4, OUT5, OUT6
      OUT1 = SINGLE
      OUT2 = DOUBLE
      OUT3 = SEMICOL
      OUT4 = EXCLA
      OUT5 = OPENPAR
      OUT6 = CLOSEPAR
      RETURN
      END
    """

    @pytest.mark.skipif(sys.platform=='win32',
                        reason='Fails with MinGW64 Gfortran (Issue #9673)')
    def test_quoted_character(self):
        assert_equal(self.module.foo(), (b"'", b'"', b';', b'!', b'(', b')'))
