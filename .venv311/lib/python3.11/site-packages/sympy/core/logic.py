"""Logic expressions handling

NOTE
----

at present this is mainly needed for facts.py, feel free however to improve
this stuff for general purpose.
"""

from __future__ import annotations
from typing import Optional

# Type of a fuzzy bool
FuzzyBool = Optional[bool]


def _torf(args):
    """Return True if all args are True, False if they
    are all False, else None.

    >>> from sympy.core.logic import _torf
    >>> _torf((True, True))
    True
    >>> _torf((False, False))
    False
    >>> _torf((True, False))
    """
    sawT = sawF = False
    for a in args:
        if a is True:
            if sawF:
                return
            sawT = True
        elif a is False:
            if sawT:
                return
            sawF = True
        else:
            return
    return sawT


def _fuzzy_group(args, quick_exit=False):
    """Return True if all args are True, None if there is any None else False
    unless ``quick_exit`` is True (then return None as soon as a second False
    is seen.

     ``_fuzzy_group`` is like ``fuzzy_and`` except that it is more
    conservative in returning a False, waiting to make sure that all
    arguments are True or False and returning None if any arguments are
    None. It also has the capability of permiting only a single False and
    returning None if more than one is seen. For example, the presence of a
    single transcendental amongst rationals would indicate that the group is
    no longer rational; but a second transcendental in the group would make the
    determination impossible.


    Examples
    ========

    >>> from sympy.core.logic import _fuzzy_group

    By default, multiple Falses mean the group is broken:

    >>> _fuzzy_group([False, False, True])
    False

    If multiple Falses mean the group status is unknown then set
    `quick_exit` to True so None can be returned when the 2nd False is seen:

    >>> _fuzzy_group([False, False, True], quick_exit=True)

    But if only a single False is seen then the group is known to
    be broken:

    >>> _fuzzy_group([False, True, True], quick_exit=True)
    False

    """
    saw_other = False
    for a in args:
        if a is True:
            continue
        if a is None:
            return
        if quick_exit and saw_other:
            return
        saw_other = True
    return not saw_other


def fuzzy_bool(x):
    """Return True, False or None according to x.

    Whereas bool(x) returns True or False, fuzzy_bool allows
    for the None value and non-false values (which become None), too.

    Examples
    ========

    >>> from sympy.core.logic import fuzzy_bool
    >>> from sympy.abc import x
    >>> fuzzy_bool(x), fuzzy_bool(None)
    (None, None)
    >>> bool(x), bool(None)
    (True, False)

    """
    if x is None:
        return None
    if x in (True, False):
        return bool(x)


def fuzzy_and(args):
    """Return True (all True), False (any False) or None.

    Examples
    ========

    >>> from sympy.core.logic import fuzzy_and
    >>> from sympy import Dummy

    If you had a list of objects to test the commutivity of
    and you want the fuzzy_and logic applied, passing an
    iterator will allow the commutativity to only be computed
    as many times as necessary. With this list, False can be
    returned after analyzing the first symbol:

    >>> syms = [Dummy(commutative=False), Dummy()]
    >>> fuzzy_and(s.is_commutative for s in syms)
    False

    That False would require less work than if a list of pre-computed
    items was sent:

    >>> fuzzy_and([s.is_commutative for s in syms])
    False
    """

    rv = True
    for ai in args:
        ai = fuzzy_bool(ai)
        if ai is False:
            return False
        if rv:  # this will stop updating if a None is ever trapped
            rv = ai
    return rv


def fuzzy_not(v):
    """
    Not in fuzzy logic

    Return None if `v` is None else `not v`.

    Examples
    ========

    >>> from sympy.core.logic import fuzzy_not
    >>> fuzzy_not(True)
    False
    >>> fuzzy_not(None)
    >>> fuzzy_not(False)
    True

    """
    if v is None:
        return v
    else:
        return not v


def fuzzy_or(args):
    """
    Or in fuzzy logic. Returns True (any True), False (all False), or None

    See the docstrings of fuzzy_and and fuzzy_not for more info.  fuzzy_or is
    related to the two by the standard De Morgan's law.

    >>> from sympy.core.logic import fuzzy_or
    >>> fuzzy_or([True, False])
    True
    >>> fuzzy_or([True, None])
    True
    >>> fuzzy_or([False, False])
    False
    >>> print(fuzzy_or([False, None]))
    None

    """
    rv = False
    for ai in args:
        ai = fuzzy_bool(ai)
        if ai is True:
            return True
        if rv is False:  # this will stop updating if a None is ever trapped
            rv = ai
    return rv


def fuzzy_xor(args):
    """Return None if any element of args is not True or False, else
    True (if there are an odd number of True elements), else False."""
    t = 0
    for a in args:
        ai = fuzzy_bool(a)
        if ai:
            t += 1
        elif ai is None:
            return
    return t % 2 == 1


def fuzzy_nand(args):
    """Return False if all args are True, True if they are all False,
    else None."""
    return fuzzy_not(fuzzy_and(args))


class Logic:
    """Logical expression"""
    # {} 'op' -> LogicClass
    op_2class: dict[str, type[Logic]] = {}

    def __new__(cls, *args):
        obj = object.__new__(cls)
        obj.args = args
        return obj

    def __getnewargs__(self):
        return self.args

    def __hash__(self):
        return hash((type(self).__name__,) + tuple(self.args))

    def __eq__(a, b):
        if not isinstance(b, type(a)):
            return False
        else:
            return a.args == b.args

    def __ne__(a, b):
        if not isinstance(b, type(a)):
            return True
        else:
            return a.args != b.args

    def __lt__(self, other):
        if self.__cmp__(other) == -1:
            return True
        return False

    def __cmp__(self, other):
        if type(self) is not type(other):
            a = str(type(self))
            b = str(type(other))
        else:
            a = self.args
            b = other.args
        return (a > b) - (a < b)

    def __str__(self):
        return '%s(%s)' % (self.__class__.__name__,
                           ', '.join(str(a) for a in self.args))

    __repr__ = __str__

    @staticmethod
    def fromstring(text):
        """Logic from string with space around & and | but none after !.

           e.g.

           !a & b | c
        """
        lexpr = None  # current logical expression
        schedop = None  # scheduled operation
        for term in text.split():
            # operation symbol
            if term in '&|':
                if schedop is not None:
                    raise ValueError(
                        'double op forbidden: "%s %s"' % (term, schedop))
                if lexpr is None:
                    raise ValueError(
                        '%s cannot be in the beginning of expression' % term)
                schedop = term
                continue
            if '&' in term or '|' in term:
                raise ValueError('& and | must have space around them')
            if term[0] == '!':
                if len(term) == 1:
                    raise ValueError('do not include space after "!"')
                term = Not(term[1:])

            # already scheduled operation, e.g. '&'
            if schedop:
                lexpr = Logic.op_2class[schedop](lexpr, term)
                schedop = None
                continue

            # this should be atom
            if lexpr is not None:
                raise ValueError(
                    'missing op between "%s" and "%s"' % (lexpr, term))

            lexpr = term

        # let's check that we ended up in correct state
        if schedop is not None:
            raise ValueError('premature end-of-expression in "%s"' % text)
        if lexpr is None:
            raise ValueError('"%s" is empty' % text)

        # everything looks good now
        return lexpr


class AndOr_Base(Logic):

    def __new__(cls, *args):
        bargs = []
        for a in args:
            if a == cls.op_x_notx:
                return a
            elif a == (not cls.op_x_notx):
                continue    # skip this argument
            bargs.append(a)

        args = sorted(set(cls.flatten(bargs)), key=hash)

        for a in args:
            if Not(a) in args:
                return cls.op_x_notx

        if len(args) == 1:
            return args.pop()
        elif len(args) == 0:
            return not cls.op_x_notx

        return Logic.__new__(cls, *args)

    @classmethod
    def flatten(cls, args):
        # quick-n-dirty flattening for And and Or
        args_queue = list(args)
        res = []

        while True:
            try:
                arg = args_queue.pop(0)
            except IndexError:
                break
            if isinstance(arg, Logic):
                if isinstance(arg, cls):
                    args_queue.extend(arg.args)
                    continue
            res.append(arg)

        args = tuple(res)
        return args


class And(AndOr_Base):
    op_x_notx = False

    def _eval_propagate_not(self):
        # !(a&b&c ...) == !a | !b | !c ...
        return Or(*[Not(a) for a in self.args])

    # (a|b|...) & c == (a&c) | (b&c) | ...
    def expand(self):

        # first locate Or
        for i, arg in enumerate(self.args):
            if isinstance(arg, Or):
                arest = self.args[:i] + self.args[i + 1:]

                orterms = [And(*(arest + (a,))) for a in arg.args]
                for j in range(len(orterms)):
                    if isinstance(orterms[j], Logic):
                        orterms[j] = orterms[j].expand()

                res = Or(*orterms)
                return res

        return self


class Or(AndOr_Base):
    op_x_notx = True

    def _eval_propagate_not(self):
        # !(a|b|c ...) == !a & !b & !c ...
        return And(*[Not(a) for a in self.args])


class Not(Logic):

    def __new__(cls, arg):
        if isinstance(arg, str):
            return Logic.__new__(cls, arg)

        elif isinstance(arg, bool):
            return not arg
        elif isinstance(arg, Not):
            return arg.args[0]

        elif isinstance(arg, Logic):
            # XXX this is a hack to expand right from the beginning
            arg = arg._eval_propagate_not()
            return arg

        else:
            raise ValueError('Not: unknown argument %r' % (arg,))

    @property
    def arg(self):
        return self.args[0]


Logic.op_2class['&'] = And
Logic.op_2class['|'] = Or
Logic.op_2class['!'] = Not
