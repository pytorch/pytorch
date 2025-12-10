from sympy.interactive.session import int_to_Integer


def test_int_to_Integer():
    assert int_to_Integer("1 + 2.2 + 0x3 + 40") == \
        'Integer (1 )+2.2 +Integer (0x3 )+Integer (40 )'
    assert int_to_Integer("0b101") == 'Integer (0b101 )'
    assert int_to_Integer("ab1 + 1 + '1 + 2'") == "ab1 +Integer (1 )+'1 + 2'"
    assert int_to_Integer("(2 + \n3)") == '(Integer (2 )+\nInteger (3 ))'
    assert int_to_Integer("2 + 2.0 + 2j + 2e-10") == 'Integer (2 )+2.0 +2j +2e-10 '
