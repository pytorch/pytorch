from numpy.testing import assert_

import numbers

import numpy as np
from numpy.core.numerictypes import sctypes

class TestABC:
    def test_abstract(self):
        assert_(issubclass(np.number, numbers.Number))

        assert_(issubclass(np.inexact, numbers.Complex))
        assert_(issubclass(np.complexfloating, numbers.Complex))
        assert_(issubclass(np.floating, numbers.Real))

        assert_(issubclass(np.integer, numbers.Integral))
        assert_(issubclass(np.signedinteger, numbers.Integral))
        assert_(issubclass(np.unsignedinteger, numbers.Integral))

    def test_floats(self):
        for t in sctypes['float']:
            assert_(isinstance(t(), numbers.Real),
                    "{0} is not instance of Real".format(t.__name__))
            assert_(issubclass(t, numbers.Real),
                    "{0} is not subclass of Real".format(t.__name__))
            assert_(not isinstance(t(), numbers.Rational),
                    "{0} is instance of Rational".format(t.__name__))
            assert_(not issubclass(t, numbers.Rational),
                    "{0} is subclass of Rational".format(t.__name__))

    def test_complex(self):
        for t in sctypes['complex']:
            assert_(isinstance(t(), numbers.Complex),
                    "{0} is not instance of Complex".format(t.__name__))
            assert_(issubclass(t, numbers.Complex),
                    "{0} is not subclass of Complex".format(t.__name__))
            assert_(not isinstance(t(), numbers.Real),
                    "{0} is instance of Real".format(t.__name__))
            assert_(not issubclass(t, numbers.Real),
                    "{0} is subclass of Real".format(t.__name__))

    def test_int(self):
        for t in sctypes['int']:
            assert_(isinstance(t(), numbers.Integral),
                    "{0} is not instance of Integral".format(t.__name__))
            assert_(issubclass(t, numbers.Integral),
                    "{0} is not subclass of Integral".format(t.__name__))

    def test_uint(self):
        for t in sctypes['uint']:
            assert_(isinstance(t(), numbers.Integral),
                    "{0} is not instance of Integral".format(t.__name__))
            assert_(issubclass(t, numbers.Integral),
                    "{0} is not subclass of Integral".format(t.__name__))
