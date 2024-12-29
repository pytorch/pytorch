"""Various tests on satisfiability using dimacs cnf file syntax
You can find lots of cnf files in
ftp://dimacs.rutgers.edu/pub/challenge/satisfiability/benchmarks/cnf/
"""

from sympy.logic.utilities.dimacs import load
from sympy.logic.algorithms.dpll import dpll_satisfiable


def test_f1():
    assert bool(dpll_satisfiable(load(f1)))


def test_f2():
    assert bool(dpll_satisfiable(load(f2)))


def test_f3():
    assert bool(dpll_satisfiable(load(f3)))


def test_f4():
    assert not bool(dpll_satisfiable(load(f4)))


def test_f5():
    assert bool(dpll_satisfiable(load(f5)))

f1 = """c  simple example
c Resolution: SATISFIABLE
c
p cnf 3 2
1 -3 0
2 3 -1 0
"""


f2 = """c  an example from Quinn's text, 16 variables and 18 clauses.
c Resolution: SATISFIABLE
c
p cnf 16 18
  1    2  0
 -2   -4  0
  3    4  0
 -4   -5  0
  5   -6  0
  6   -7  0
  6    7  0
  7  -16  0
  8   -9  0
 -8  -14  0
  9   10  0
  9  -10  0
-10  -11  0
 10   12  0
 11   12  0
 13   14  0
 14  -15  0
 15   16  0
"""

f3 = """c
p cnf 6 9
-1 0
-3 0
2 -1 0
2 -4 0
5 -4 0
-1 -3 0
-4 -6 0
1 3 -2 0
4 6 -2 -5 0
"""

f4 = """c
c file:   hole6.cnf [http://people.sc.fsu.edu/~jburkardt/data/cnf/hole6.cnf]
c
c SOURCE: John Hooker (jh38+@andrew.cmu.edu)
c
c DESCRIPTION: Pigeon hole problem of placing n (for file 'holen.cnf') pigeons
c              in n+1 holes without placing 2 pigeons in the same hole
c
c NOTE: Part of the collection at the Forschungsinstitut fuer
c       anwendungsorientierte Wissensverarbeitung in Ulm Germany.
c
c NOTE: Not satisfiable
c
p cnf 42 133
-1     -7    0
-1     -13   0
-1     -19   0
-1     -25   0
-1     -31   0
-1     -37   0
-7     -13   0
-7     -19   0
-7     -25   0
-7     -31   0
-7     -37   0
-13    -19   0
-13    -25   0
-13    -31   0
-13    -37   0
-19    -25   0
-19    -31   0
-19    -37   0
-25    -31   0
-25    -37   0
-31    -37   0
-2     -8    0
-2     -14   0
-2     -20   0
-2     -26   0
-2     -32   0
-2     -38   0
-8     -14   0
-8     -20   0
-8     -26   0
-8     -32   0
-8     -38   0
-14    -20   0
-14    -26   0
-14    -32   0
-14    -38   0
-20    -26   0
-20    -32   0
-20    -38   0
-26    -32   0
-26    -38   0
-32    -38   0
-3     -9    0
-3     -15   0
-3     -21   0
-3     -27   0
-3     -33   0
-3     -39   0
-9     -15   0
-9     -21   0
-9     -27   0
-9     -33   0
-9     -39   0
-15    -21   0
-15    -27   0
-15    -33   0
-15    -39   0
-21    -27   0
-21    -33   0
-21    -39   0
-27    -33   0
-27    -39   0
-33    -39   0
-4     -10   0
-4     -16   0
-4     -22   0
-4     -28   0
-4     -34   0
-4     -40   0
-10    -16   0
-10    -22   0
-10    -28   0
-10    -34   0
-10    -40   0
-16    -22   0
-16    -28   0
-16    -34   0
-16    -40   0
-22    -28   0
-22    -34   0
-22    -40   0
-28    -34   0
-28    -40   0
-34    -40   0
-5     -11   0
-5     -17   0
-5     -23   0
-5     -29   0
-5     -35   0
-5     -41   0
-11    -17   0
-11    -23   0
-11    -29   0
-11    -35   0
-11    -41   0
-17    -23   0
-17    -29   0
-17    -35   0
-17    -41   0
-23    -29   0
-23    -35   0
-23    -41   0
-29    -35   0
-29    -41   0
-35    -41   0
-6     -12   0
-6     -18   0
-6     -24   0
-6     -30   0
-6     -36   0
-6     -42   0
-12    -18   0
-12    -24   0
-12    -30   0
-12    -36   0
-12    -42   0
-18    -24   0
-18    -30   0
-18    -36   0
-18    -42   0
-24    -30   0
-24    -36   0
-24    -42   0
-30    -36   0
-30    -42   0
-36    -42   0
 6      5      4      3      2      1    0
 12     11     10     9      8      7    0
 18     17     16     15     14     13   0
 24     23     22     21     20     19   0
 30     29     28     27     26     25   0
 36     35     34     33     32     31   0
 42     41     40     39     38     37   0
"""

f5 = """c  simple example requiring variable selection
c
c NOTE: Satisfiable
c
p cnf 5 5
1 2 3 0
1 -2 3 0
4 5 -3 0
1 -4 -3 0
-1 -5 0
"""
