from sympy.combinatorics.graycode import (GrayCode, bin_to_gray,
    random_bitstring, get_subset_from_bitstring, graycode_subsets,
    gray_to_bin)
from sympy.testing.pytest import raises

def test_graycode():
    g = GrayCode(2)
    got = []
    for i in g.generate_gray():
        if i.startswith('0'):
            g.skip()
        got.append(i)
    assert got == '00 11 10'.split()
    a = GrayCode(6)
    assert a.current == '0'*6
    assert a.rank == 0
    assert len(list(a.generate_gray())) == 64
    codes = ['011001', '011011', '011010',
    '011110', '011111', '011101', '011100', '010100', '010101', '010111',
    '010110', '010010', '010011', '010001', '010000', '110000', '110001',
    '110011', '110010', '110110', '110111', '110101', '110100', '111100',
    '111101', '111111', '111110', '111010', '111011', '111001', '111000',
    '101000', '101001', '101011', '101010', '101110', '101111', '101101',
    '101100', '100100', '100101', '100111', '100110', '100010', '100011',
    '100001', '100000']
    assert list(a.generate_gray(start='011001')) == codes
    assert list(
        a.generate_gray(rank=GrayCode(6, start='011001').rank)) == codes
    assert a.next().current == '000001'
    assert a.next(2).current == '000011'
    assert a.next(-1).current == '100000'

    a = GrayCode(5, start='10010')
    assert a.rank == 28
    a = GrayCode(6, start='101000')
    assert a.rank == 48

    assert GrayCode(6, rank=4).current == '000110'
    assert GrayCode(6, rank=4).rank == 4
    assert [GrayCode(4, start=s).rank for s in
            GrayCode(4).generate_gray()] == [0, 1, 2, 3, 4, 5, 6, 7, 8,
                                             9, 10, 11, 12, 13, 14, 15]
    a = GrayCode(15, rank=15)
    assert a.current == '000000000001000'

    assert bin_to_gray('111') == '100'

    a = random_bitstring(5)
    assert type(a) is str
    assert len(a) == 5
    assert all(i in ['0', '1'] for i in a)

    assert get_subset_from_bitstring(
        ['a', 'b', 'c', 'd'], '0011') == ['c', 'd']
    assert get_subset_from_bitstring('abcd', '1001') == ['a', 'd']
    assert list(graycode_subsets(['a', 'b', 'c'])) == \
        [[], ['c'], ['b', 'c'], ['b'], ['a', 'b'], ['a', 'b', 'c'],
         ['a', 'c'], ['a']]

    raises(ValueError, lambda: GrayCode(0))
    raises(ValueError, lambda: GrayCode(2.2))
    raises(ValueError, lambda: GrayCode(2, start=[1, 1, 0]))
    raises(ValueError, lambda: GrayCode(2, rank=2.5))
    raises(ValueError, lambda: get_subset_from_bitstring(['c', 'a', 'c'], '1100'))
    raises(ValueError, lambda: list(GrayCode(3).generate_gray(start="1111")))


def test_live_issue_117():
    assert bin_to_gray('0100') == '0110'
    assert bin_to_gray('0101') == '0111'
    for bits in ('0100', '0101'):
        assert gray_to_bin(bin_to_gray(bits)) == bits
