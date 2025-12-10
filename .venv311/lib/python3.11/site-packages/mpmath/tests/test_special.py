from mpmath import *

def test_special():
    assert inf == inf
    assert inf != -inf
    assert -inf == -inf
    assert inf != nan
    assert nan != nan
    assert isnan(nan)
    assert --inf == inf
    assert abs(inf) == inf
    assert abs(-inf) == inf
    assert abs(nan) != abs(nan)

    assert isnan(inf - inf)
    assert isnan(inf + (-inf))
    assert isnan(-inf - (-inf))

    assert isnan(inf + nan)
    assert isnan(-inf + nan)

    assert mpf(2) + inf == inf
    assert 2 + inf == inf
    assert mpf(2) - inf == -inf
    assert 2 - inf == -inf

    assert inf > 3
    assert 3 < inf
    assert 3 > -inf
    assert -inf < 3
    assert inf > mpf(3)
    assert mpf(3) < inf
    assert mpf(3) > -inf
    assert -inf < mpf(3)

    assert not (nan < 3)
    assert not (nan > 3)

    assert isnan(inf * 0)
    assert isnan(-inf * 0)
    assert inf * 3 == inf
    assert inf * -3 == -inf
    assert -inf * 3 == -inf
    assert -inf * -3 == inf
    assert inf * inf == inf
    assert -inf * -inf == inf

    assert isnan(nan / 3)
    assert inf / -3 == -inf
    assert inf / 3 == inf
    assert 3 / inf == 0
    assert -3 / inf == 0
    assert 0 / inf == 0
    assert isnan(inf / inf)
    assert isnan(inf / -inf)
    assert isnan(inf / nan)

    assert mpf('inf') == mpf('+inf') == inf
    assert mpf('-inf') == -inf
    assert isnan(mpf('nan'))

    assert isinf(inf)
    assert isinf(-inf)
    assert not isinf(mpf(0))
    assert not isinf(nan)

def test_special_powers():
    assert inf**3 == inf
    assert isnan(inf**0)
    assert inf**-3 == 0
    assert (-inf)**2 == inf
    assert (-inf)**3 == -inf
    assert isnan((-inf)**0)
    assert (-inf)**-2 == 0
    assert (-inf)**-3 == 0
    assert isnan(nan**5)
    assert isnan(nan**0)

def test_functions_special():
    assert exp(inf) == inf
    assert exp(-inf) == 0
    assert isnan(exp(nan))
    assert log(inf) == inf
    assert isnan(log(nan))
    assert isnan(sin(inf))
    assert isnan(sin(nan))
    assert atan(inf).ae(pi/2)
    assert atan(-inf).ae(-pi/2)
    assert isnan(sqrt(nan))
    assert sqrt(inf) == inf

def test_convert_special():
    float_inf = 1e300 * 1e300
    float_ninf = -float_inf
    float_nan = float_inf/float_ninf
    assert mpf(3) * float_inf == inf
    assert mpf(3) * float_ninf == -inf
    assert isnan(mpf(3) * float_nan)
    assert not (mpf(3) < float_nan)
    assert not (mpf(3) > float_nan)
    assert not (mpf(3) <= float_nan)
    assert not (mpf(3) >= float_nan)
    assert float(mpf('1e1000')) == float_inf
    assert float(mpf('-1e1000')) == float_ninf
    assert float(mpf('1e100000000000000000')) == float_inf
    assert float(mpf('-1e100000000000000000')) == float_ninf
    assert float(mpf('1e-100000000000000000')) == 0.0

def test_div_bug():
    assert isnan(nan/1)
    assert isnan(nan/2)
    assert inf/2 == inf
    assert (-inf)/2 == -inf
