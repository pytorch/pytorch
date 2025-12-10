"""Tests for PythonRational type. """

from sympy.polys.domains import PythonRational as QQ
from sympy.testing.pytest import raises

def test_PythonRational__init__():
    assert QQ(0).numerator == 0
    assert QQ(0).denominator == 1
    assert QQ(0, 1).numerator == 0
    assert QQ(0, 1).denominator == 1
    assert QQ(0, -1).numerator == 0
    assert QQ(0, -1).denominator == 1

    assert QQ(1).numerator == 1
    assert QQ(1).denominator == 1
    assert QQ(1, 1).numerator == 1
    assert QQ(1, 1).denominator == 1
    assert QQ(-1, -1).numerator == 1
    assert QQ(-1, -1).denominator == 1

    assert QQ(-1).numerator == -1
    assert QQ(-1).denominator == 1
    assert QQ(-1, 1).numerator == -1
    assert QQ(-1, 1).denominator == 1
    assert QQ( 1, -1).numerator == -1
    assert QQ( 1, -1).denominator == 1

    assert QQ(1, 2).numerator == 1
    assert QQ(1, 2).denominator == 2
    assert QQ(3, 4).numerator == 3
    assert QQ(3, 4).denominator == 4

    assert QQ(2, 2).numerator == 1
    assert QQ(2, 2).denominator == 1
    assert QQ(2, 4).numerator == 1
    assert QQ(2, 4).denominator == 2

def test_PythonRational__hash__():
    assert hash(QQ(0)) == hash(0)
    assert hash(QQ(1)) == hash(1)
    assert hash(QQ(117)) == hash(117)

def test_PythonRational__int__():
    assert int(QQ(-1, 4)) == 0
    assert int(QQ( 1, 4)) == 0
    assert int(QQ(-5, 4)) == -1
    assert int(QQ( 5, 4)) == 1

def test_PythonRational__float__():
    assert float(QQ(-1, 2)) == -0.5
    assert float(QQ( 1, 2)) == 0.5

def test_PythonRational__abs__():
    assert abs(QQ(-1, 2)) == QQ(1, 2)
    assert abs(QQ( 1, 2)) == QQ(1, 2)

def test_PythonRational__pos__():
    assert +QQ(-1, 2) == QQ(-1, 2)
    assert +QQ( 1, 2) == QQ( 1, 2)

def test_PythonRational__neg__():
    assert -QQ(-1, 2) == QQ( 1, 2)
    assert -QQ( 1, 2) == QQ(-1, 2)

def test_PythonRational__add__():
    assert QQ(-1, 2) + QQ( 1, 2) == QQ(0)
    assert QQ( 1, 2) + QQ(-1, 2) == QQ(0)

    assert QQ(1, 2) + QQ(1, 2) == QQ(1)
    assert QQ(1, 2) + QQ(3, 2) == QQ(2)
    assert QQ(3, 2) + QQ(1, 2) == QQ(2)
    assert QQ(3, 2) + QQ(3, 2) == QQ(3)

    assert 1 + QQ(1, 2) == QQ(3, 2)
    assert QQ(1, 2) + 1 == QQ(3, 2)

def test_PythonRational__sub__():
    assert QQ(-1, 2) - QQ( 1, 2) == QQ(-1)
    assert QQ( 1, 2) - QQ(-1, 2) == QQ( 1)

    assert QQ(1, 2) - QQ(1, 2) == QQ( 0)
    assert QQ(1, 2) - QQ(3, 2) == QQ(-1)
    assert QQ(3, 2) - QQ(1, 2) == QQ( 1)
    assert QQ(3, 2) - QQ(3, 2) == QQ( 0)

    assert 1 - QQ(1, 2) == QQ( 1, 2)
    assert QQ(1, 2) - 1 == QQ(-1, 2)

def test_PythonRational__mul__():
    assert QQ(-1, 2) * QQ( 1, 2) == QQ(-1, 4)
    assert QQ( 1, 2) * QQ(-1, 2) == QQ(-1, 4)

    assert QQ(1, 2) * QQ(1, 2) == QQ(1, 4)
    assert QQ(1, 2) * QQ(3, 2) == QQ(3, 4)
    assert QQ(3, 2) * QQ(1, 2) == QQ(3, 4)
    assert QQ(3, 2) * QQ(3, 2) == QQ(9, 4)

    assert 2 * QQ(1, 2) == QQ(1)
    assert QQ(1, 2) * 2 == QQ(1)

def test_PythonRational__truediv__():
    assert QQ(-1, 2) / QQ( 1, 2) == QQ(-1)
    assert QQ( 1, 2) / QQ(-1, 2) == QQ(-1)

    assert QQ(1, 2) / QQ(1, 2) == QQ(1)
    assert QQ(1, 2) / QQ(3, 2) == QQ(1, 3)
    assert QQ(3, 2) / QQ(1, 2) == QQ(3)
    assert QQ(3, 2) / QQ(3, 2) == QQ(1)

    assert 2 / QQ(1, 2) == QQ(4)
    assert QQ(1, 2) / 2 == QQ(1, 4)

    raises(ZeroDivisionError, lambda: QQ(1, 2) / QQ(0))
    raises(ZeroDivisionError, lambda: QQ(1, 2) / 0)

def test_PythonRational__pow__():
    assert QQ(1)**10 == QQ(1)
    assert QQ(2)**10 == QQ(1024)

    assert QQ(1)**(-10) == QQ(1)
    assert QQ(2)**(-10) == QQ(1, 1024)

def test_PythonRational__eq__():
    assert (QQ(1, 2) == QQ(1, 2)) is True
    assert (QQ(1, 2) != QQ(1, 2)) is False

    assert (QQ(1, 2) == QQ(1, 3)) is False
    assert (QQ(1, 2) != QQ(1, 3)) is True

def test_PythonRational__lt_le_gt_ge__():
    assert (QQ(1, 2) < QQ(1, 4)) is False
    assert (QQ(1, 2) <= QQ(1, 4)) is False
    assert (QQ(1, 2) > QQ(1, 4)) is True
    assert (QQ(1, 2) >= QQ(1, 4)) is True

    assert (QQ(1, 4) < QQ(1, 2)) is True
    assert (QQ(1, 4) <= QQ(1, 2)) is True
    assert (QQ(1, 4) > QQ(1, 2)) is False
    assert (QQ(1, 4) >= QQ(1, 2)) is False
