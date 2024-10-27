"""For reading in DIMACS file format

www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/satformat.ps

"""

from sympy.core import Symbol
from sympy.logic.boolalg import And, Or
import re


def load(s):
    """Loads a boolean expression from a string.

    Examples
    ========

    >>> from sympy.logic.utilities.dimacs import load
    >>> load('1')
    cnf_1
    >>> load('1 2')
    cnf_1 | cnf_2
    >>> load('1 \\n 2')
    cnf_1 & cnf_2
    >>> load('1 2 \\n 3')
    cnf_3 & (cnf_1 | cnf_2)
    """
    clauses = []

    lines = s.split('\n')

    pComment = re.compile(r'c.*')
    pStats = re.compile(r'p\s*cnf\s*(\d*)\s*(\d*)')

    while len(lines) > 0:
        line = lines.pop(0)

        # Only deal with lines that aren't comments
        if not pComment.match(line):
            m = pStats.match(line)

            if not m:
                nums = line.rstrip('\n').split(' ')
                list = []
                for lit in nums:
                    if lit != '':
                        if int(lit) == 0:
                            continue
                        num = abs(int(lit))
                        sign = True
                        if int(lit) < 0:
                            sign = False

                        if sign:
                            list.append(Symbol("cnf_%s" % num))
                        else:
                            list.append(~Symbol("cnf_%s" % num))

                if len(list) > 0:
                    clauses.append(Or(*list))

    return And(*clauses)


def load_file(location):
    """Loads a boolean expression from a file."""
    with open(location) as f:
        s = f.read()

    return load(s)
