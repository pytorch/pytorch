"""
This module contains the machinery handling assumptions.
Do also consider the guide :ref:`assumptions-guide`.

All symbolic objects have assumption attributes that can be accessed via
``.is_<assumption name>`` attribute.

Assumptions determine certain properties of symbolic objects and can
have 3 possible values: ``True``, ``False``, ``None``.  ``True`` is returned if the
object has the property and ``False`` is returned if it does not or cannot
(i.e. does not make sense):

    >>> from sympy import I
    >>> I.is_algebraic
    True
    >>> I.is_real
    False
    >>> I.is_prime
    False

When the property cannot be determined (or when a method is not
implemented) ``None`` will be returned. For example,  a generic symbol, ``x``,
may or may not be positive so a value of ``None`` is returned for ``x.is_positive``.

By default, all symbolic values are in the largest set in the given context
without specifying the property. For example, a symbol that has a property
being integer, is also real, complex, etc.

Here follows a list of possible assumption names:

.. glossary::

    commutative
        object commutes with any other object with
        respect to multiplication operation. See [12]_.

    complex
        object can have only values from the set
        of complex numbers. See [13]_.

    imaginary
        object value is a number that can be written as a real
        number multiplied by the imaginary unit ``I``.  See
        [3]_.  Please note that ``0`` is not considered to be an
        imaginary number, see
        `issue #7649 <https://github.com/sympy/sympy/issues/7649>`_.

    real
        object can have only values from the set
        of real numbers.

    extended_real
        object can have only values from the set
        of real numbers, ``oo`` and ``-oo``.

    integer
        object can have only values from the set
        of integers.

    odd
    even
        object can have only values from the set of
        odd (even) integers [2]_.

    prime
        object is a natural number greater than 1 that has
        no positive divisors other than 1 and itself.  See [6]_.

    composite
        object is a positive integer that has at least one positive
        divisor other than 1 or the number itself.  See [4]_.

    zero
        object has the value of 0.

    nonzero
        object is a real number that is not zero.

    rational
        object can have only values from the set
        of rationals.

    algebraic
        object can have only values from the set
        of algebraic numbers [11]_.

    transcendental
        object can have only values from the set
        of transcendental numbers [10]_.

    irrational
        object value cannot be represented exactly by :class:`~.Rational`, see [5]_.

    finite
    infinite
        object absolute value is bounded (arbitrarily large).
        See [7]_, [8]_, [9]_.

    negative
    nonnegative
        object can have only negative (nonnegative)
        values [1]_.

    positive
    nonpositive
        object can have only positive (nonpositive) values.

    extended_negative
    extended_nonnegative
    extended_positive
    extended_nonpositive
    extended_nonzero
        as without the extended part, but also including infinity with
        corresponding sign, e.g., extended_positive includes ``oo``

    hermitian
    antihermitian
        object belongs to the field of Hermitian
        (antihermitian) operators.

Examples
========

    >>> from sympy import Symbol
    >>> x = Symbol('x', real=True); x
    x
    >>> x.is_real
    True
    >>> x.is_complex
    True

See Also
========

.. seealso::

    :py:class:`sympy.core.numbers.ImaginaryUnit`
    :py:class:`sympy.core.numbers.Zero`
    :py:class:`sympy.core.numbers.One`
    :py:class:`sympy.core.numbers.Infinity`
    :py:class:`sympy.core.numbers.NegativeInfinity`
    :py:class:`sympy.core.numbers.ComplexInfinity`

Notes
=====

The fully-resolved assumptions for any SymPy expression
can be obtained as follows:

    >>> from sympy.core.assumptions import assumptions
    >>> x = Symbol('x',positive=True)
    >>> assumptions(x + I)
    {'commutative': True, 'complex': True, 'composite': False, 'even':
    False, 'extended_negative': False, 'extended_nonnegative': False,
    'extended_nonpositive': False, 'extended_nonzero': False,
    'extended_positive': False, 'extended_real': False, 'finite': True,
    'imaginary': False, 'infinite': False, 'integer': False, 'irrational':
    False, 'negative': False, 'noninteger': False, 'nonnegative': False,
    'nonpositive': False, 'nonzero': False, 'odd': False, 'positive':
    False, 'prime': False, 'rational': False, 'real': False, 'zero':
    False}

Developers Notes
================

The current (and possibly incomplete) values are stored
in the ``obj._assumptions dictionary``; queries to getter methods
(with property decorators) or attributes of objects/classes
will return values and update the dictionary.

    >>> eq = x**2 + I
    >>> eq._assumptions
    {}
    >>> eq.is_finite
    True
    >>> eq._assumptions
    {'finite': True, 'infinite': False}

For a :class:`~.Symbol`, there are two locations for assumptions that may
be of interest. The ``assumptions0`` attribute gives the full set of
assumptions derived from a given set of initial assumptions. The
latter assumptions are stored as ``Symbol._assumptions_orig``

    >>> Symbol('x', prime=True, even=True)._assumptions_orig
    {'even': True, 'prime': True}

The ``_assumptions_orig`` are not necessarily canonical nor are they filtered
in any way: they records the assumptions used to instantiate a Symbol and (for
storage purposes) represent a more compact representation of the assumptions
needed to recreate the full set in ``Symbol.assumptions0``.


References
==========

.. [1] https://en.wikipedia.org/wiki/Negative_number
.. [2] https://en.wikipedia.org/wiki/Parity_%28mathematics%29
.. [3] https://en.wikipedia.org/wiki/Imaginary_number
.. [4] https://en.wikipedia.org/wiki/Composite_number
.. [5] https://en.wikipedia.org/wiki/Irrational_number
.. [6] https://en.wikipedia.org/wiki/Prime_number
.. [7] https://en.wikipedia.org/wiki/Finite
.. [8] https://docs.python.org/3/library/math.html#math.isfinite
.. [9] https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
.. [10] https://en.wikipedia.org/wiki/Transcendental_number
.. [11] https://en.wikipedia.org/wiki/Algebraic_number
.. [12] https://en.wikipedia.org/wiki/Commutative_property
.. [13] https://en.wikipedia.org/wiki/Complex_number

"""

from sympy.utilities.exceptions import sympy_deprecation_warning

from .facts import FactRules, FactKB
from .sympify import sympify

from sympy.core.random import _assumptions_shuffle as shuffle
from sympy.core.assumptions_generated import generated_assumptions as _assumptions

def _load_pre_generated_assumption_rules() -> FactRules:
    """ Load the assumption rules from pre-generated data

    To update the pre-generated data, see :method::`_generate_assumption_rules`
    """
    _assume_rules=FactRules._from_python(_assumptions)
    return _assume_rules

def _generate_assumption_rules():
    """ Generate the default assumption rules

    This method should only be called to update the pre-generated
    assumption rules.

    To update the pre-generated assumptions run: bin/ask_update.py

    """
    _assume_rules = FactRules([

    'integer        ->  rational',
    'rational       ->  real',
    'rational       ->  algebraic',
    'algebraic      ->  complex',
    'transcendental ==  complex & !algebraic',
    'real           ->  hermitian',
    'imaginary      ->  complex',
    'imaginary      ->  antihermitian',
    'extended_real  ->  commutative',
    'complex        ->  commutative',
    'complex        ->  finite',

    'odd            ==  integer & !even',
    'even           ==  integer & !odd',

    'real           ->  complex',
    'extended_real  ->  real | infinite',
    'real           ==  extended_real & finite',

    'extended_real        ==  extended_negative | zero | extended_positive',
    'extended_negative    ==  extended_nonpositive & extended_nonzero',
    'extended_positive    ==  extended_nonnegative & extended_nonzero',

    'extended_nonpositive ==  extended_real & !extended_positive',
    'extended_nonnegative ==  extended_real & !extended_negative',

    'real           ==  negative | zero | positive',
    'negative       ==  nonpositive & nonzero',
    'positive       ==  nonnegative & nonzero',

    'nonpositive    ==  real & !positive',
    'nonnegative    ==  real & !negative',

    'positive       ==  extended_positive & finite',
    'negative       ==  extended_negative & finite',
    'nonpositive    ==  extended_nonpositive & finite',
    'nonnegative    ==  extended_nonnegative & finite',
    'nonzero        ==  extended_nonzero & finite',

    'zero           ->  even & finite',
    'zero           ==  extended_nonnegative & extended_nonpositive',
    'zero           ==  nonnegative & nonpositive',
    'nonzero        ->  real',

    'prime          ->  integer & positive',
    'composite      ->  integer & positive & !prime',
    '!composite     ->  !positive | !even | prime',

    'irrational     ==  real & !rational',

    'imaginary      ->  !extended_real',

    'infinite       ==  !finite',
    'noninteger     ==  extended_real & !integer',
    'extended_nonzero == extended_real & !zero',
    ])
    return _assume_rules


_assume_rules = _load_pre_generated_assumption_rules()
_assume_defined = _assume_rules.defined_facts.copy()
_assume_defined.add('polar')
_assume_defined = frozenset(_assume_defined)


def assumptions(expr, _check=None):
    """return the T/F assumptions of ``expr``"""
    n = sympify(expr)
    if n.is_Symbol:
        rv = n.assumptions0  # are any important ones missing?
        if _check is not None:
            rv = {k: rv[k] for k in set(rv) & set(_check)}
        return rv
    rv = {}
    for k in _assume_defined if _check is None else _check:
        v = getattr(n, 'is_{}'.format(k))
        if v is not None:
            rv[k] = v
    return rv


def common_assumptions(exprs, check=None):
    """return those assumptions which have the same True or False
    value for all the given expressions.

    Examples
    ========

    >>> from sympy.core import common_assumptions
    >>> from sympy import oo, pi, sqrt
    >>> common_assumptions([-4, 0, sqrt(2), 2, pi, oo])
    {'commutative': True, 'composite': False,
    'extended_real': True, 'imaginary': False, 'odd': False}

    By default, all assumptions are tested; pass an iterable of the
    assumptions to limit those that are reported:

    >>> common_assumptions([0, 1, 2], ['positive', 'integer'])
    {'integer': True}
    """
    check = _assume_defined if check is None else set(check)
    if not check or not exprs:
        return {}

    # get all assumptions for each
    assume = [assumptions(i, _check=check) for i in sympify(exprs)]
    # focus on those of interest that are True
    for i, e in enumerate(assume):
        assume[i] = {k: e[k] for k in set(e) & check}
    # what assumptions are in common?
    common = set.intersection(*[set(i) for i in assume])
    # which ones hold the same value
    a = assume[0]
    return {k: a[k] for k in common if all(a[k] == b[k]
        for b in assume)}


def failing_assumptions(expr, **assumptions):
    """
    Return a dictionary containing assumptions with values not
    matching those of the passed assumptions.

    Examples
    ========

    >>> from sympy import failing_assumptions, Symbol

    >>> x = Symbol('x', positive=True)
    >>> y = Symbol('y')
    >>> failing_assumptions(6*x + y, positive=True)
    {'positive': None}

    >>> failing_assumptions(x**2 - 1, positive=True)
    {'positive': None}

    If *expr* satisfies all of the assumptions, an empty dictionary is returned.

    >>> failing_assumptions(x**2, positive=True)
    {}

    """
    expr = sympify(expr)
    failed = {}
    for k in assumptions:
        test = getattr(expr, 'is_%s' % k, None)
        if test is not assumptions[k]:
            failed[k] = test
    return failed  # {} or {assumption: value != desired}


def check_assumptions(expr, against=None, **assume):
    """
    Checks whether assumptions of ``expr`` match the T/F assumptions
    given (or possessed by ``against``). True is returned if all
    assumptions match; False is returned if there is a mismatch and
    the assumption in ``expr`` is not None; else None is returned.

    Explanation
    ===========

    *assume* is a dict of assumptions with True or False values

    Examples
    ========

    >>> from sympy import Symbol, pi, I, exp, check_assumptions
    >>> check_assumptions(-5, integer=True)
    True
    >>> check_assumptions(pi, real=True, integer=False)
    True
    >>> check_assumptions(pi, negative=True)
    False
    >>> check_assumptions(exp(I*pi/7), real=False)
    True
    >>> x = Symbol('x', positive=True)
    >>> check_assumptions(2*x + 1, positive=True)
    True
    >>> check_assumptions(-2*x - 5, positive=True)
    False

    To check assumptions of *expr* against another variable or expression,
    pass the expression or variable as ``against``.

    >>> check_assumptions(2*x + 1, x)
    True

    To see if a number matches the assumptions of an expression, pass
    the number as the first argument, else its specific assumptions
    may not have a non-None value in the expression:

    >>> check_assumptions(x, 3)
    >>> check_assumptions(3, x)
    True

    ``None`` is returned if ``check_assumptions()`` could not conclude.

    >>> check_assumptions(2*x - 1, x)

    >>> z = Symbol('z')
    >>> check_assumptions(z, real=True)

    See Also
    ========

    failing_assumptions

    """
    expr = sympify(expr)
    if against is not None:
        if assume:
            raise ValueError(
                'Expecting `against` or `assume`, not both.')
        assume = assumptions(against)
    known = True
    for k, v in assume.items():
        if v is None:
            continue
        e = getattr(expr, 'is_' + k, None)
        if e is None:
            known = None
        elif v != e:
            return False
    return known


class StdFactKB(FactKB):
    """A FactKB specialized for the built-in rules

    This is the only kind of FactKB that Basic objects should use.
    """
    def __init__(self, facts=None):
        super().__init__(_assume_rules)
        # save a copy of the facts dict
        if not facts:
            self._generator = {}
        elif not isinstance(facts, FactKB):
            self._generator = facts.copy()
        else:
            self._generator = facts.generator
        if facts:
            self.deduce_all_facts(facts)

    def copy(self):
        return self.__class__(self)

    @property
    def generator(self):
        return self._generator.copy()


def as_property(fact):
    """Convert a fact name to the name of the corresponding property"""
    return 'is_%s' % fact


def make_property(fact):
    """Create the automagic property corresponding to a fact."""

    def getit(self):
        try:
            return self._assumptions[fact]
        except KeyError:
            if self._assumptions is self.default_assumptions:
                self._assumptions = self.default_assumptions.copy()
            return _ask(fact, self)

    getit.func_name = as_property(fact)
    return property(getit)


def _ask(fact, obj):
    """
    Find the truth value for a property of an object.

    This function is called when a request is made to see what a fact
    value is.

    For this we use several techniques:

    First, the fact-evaluation function is tried, if it exists (for
    example _eval_is_integer). Then we try related facts. For example

        rational   -->   integer

    another example is joined rule:

        integer & !odd  --> even

    so in the latter case if we are looking at what 'even' value is,
    'integer' and 'odd' facts will be asked.

    In all cases, when we settle on some fact value, its implications are
    deduced, and the result is cached in ._assumptions.
    """
    # FactKB which is dict-like and maps facts to their known values:
    assumptions = obj._assumptions

    # A dict that maps facts to their handlers:
    handler_map = obj._prop_handler

    # This is our queue of facts to check:
    facts_to_check = [fact]
    facts_queued = {fact}

    # Loop over the queue as it extends
    for fact_i in facts_to_check:

        # If fact_i has already been determined then we don't need to rerun the
        # handler. There is a potential race condition for multithreaded code
        # though because it's possible that fact_i was checked in another
        # thread. The main logic of the loop below would potentially skip
        # checking assumptions[fact] in this case so we check it once after the
        # loop to be sure.
        if fact_i in assumptions:
            continue

        # Now we call the associated handler for fact_i if it exists.
        fact_i_value = None
        handler_i = handler_map.get(fact_i)
        if handler_i is not None:
            fact_i_value = handler_i(obj)

        # If we get a new value for fact_i then we should update our knowledge
        # of fact_i as well as any related facts that can be inferred using the
        # inference rules connecting the fact_i and any other fact values that
        # are already known.
        if fact_i_value is not None:
            assumptions.deduce_all_facts(((fact_i, fact_i_value),))

        # Usually if assumptions[fact] is now not None then that is because of
        # the call to deduce_all_facts above. The handler for fact_i returned
        # True or False and knowing fact_i (which is equal to fact in the first
        # iteration) implies knowing a value for fact. It is also possible
        # though that independent code e.g. called indirectly by the handler or
        # called in another thread in a multithreaded context might have
        # resulted in assumptions[fact] being set. Either way we return it.
        fact_value = assumptions.get(fact)
        if fact_value is not None:
            return fact_value

        # Extend the queue with other facts that might determine fact_i. Here
        # we randomise the order of the facts that are checked. This should not
        # lead to any non-determinism if all handlers are logically consistent
        # with the inference rules for the facts. Non-deterministic assumptions
        # queries can result from bugs in the handlers that are exposed by this
        # call to shuffle. These are pushed to the back of the queue meaning
        # that the inference graph is traversed in breadth-first order.
        new_facts_to_check = list(_assume_rules.prereq[fact_i] - facts_queued)
        shuffle(new_facts_to_check)
        facts_to_check.extend(new_facts_to_check)
        facts_queued.update(new_facts_to_check)

    # The above loop should be able to handle everything fine in a
    # single-threaded context but in multithreaded code it is possible that
    # this thread skipped computing a particular fact that was computed in
    # another thread (due to the continue). In that case it is possible that
    # fact was inferred and is now stored in the assumptions dict but it wasn't
    # checked for in the body of the loop. This is an obscure case but to make
    # sure we catch it we check once here at the end of the loop.
    if fact in assumptions:
        return assumptions[fact]

    # This query can not be answered. It's possible that e.g. another thread
    # has already stored None for fact but assumptions._tell does not mind if
    # we call _tell twice setting the same value. If this raises
    # InconsistentAssumptions then it probably means that another thread
    # attempted to compute this and got a value of True or False rather than
    # None. In that case there must be a bug in at least one of the handlers.
    # If the handlers are all deterministic and are consistent with the
    # inference rules then the same value should be computed for fact in all
    # threads.
    assumptions._tell(fact, None)
    return None


def _prepare_class_assumptions(cls):
    """Precompute class level assumptions and generate handlers.

    This is called by Basic.__init_subclass__ each time a Basic subclass is
    defined.
    """

    local_defs = {}
    for k in _assume_defined:
        attrname = as_property(k)
        v = cls.__dict__.get(attrname, '')
        if isinstance(v, (bool, int, type(None))):
            if v is not None:
                v = bool(v)
            local_defs[k] = v

    defs = {}
    for base in reversed(cls.__bases__):
        assumptions = getattr(base, '_explicit_class_assumptions', None)
        if assumptions is not None:
            defs.update(assumptions)
    defs.update(local_defs)

    cls._explicit_class_assumptions = defs
    cls.default_assumptions = StdFactKB(defs)

    cls._prop_handler = {}
    for k in _assume_defined:
        eval_is_meth = getattr(cls, '_eval_is_%s' % k, None)
        if eval_is_meth is not None:
            cls._prop_handler[k] = eval_is_meth

    # Put definite results directly into the class dict, for speed
    for k, v in cls.default_assumptions.items():
        setattr(cls, as_property(k), v)

    # protection e.g. for Integer.is_even=F <- (Rational.is_integer=F)
    derived_from_bases = set()
    for base in cls.__bases__:
        default_assumptions = getattr(base, 'default_assumptions', None)
        # is an assumption-aware class
        if default_assumptions is not None:
            derived_from_bases.update(default_assumptions)

    for fact in derived_from_bases - set(cls.default_assumptions):
        pname = as_property(fact)
        if pname not in cls.__dict__:
            setattr(cls, pname, make_property(fact))

    # Finally, add any missing automagic property (e.g. for Basic)
    for fact in _assume_defined:
        pname = as_property(fact)
        if not hasattr(cls, pname):
            setattr(cls, pname, make_property(fact))


# XXX: ManagedProperties used to be the metaclass for Basic but now Basic does
# not use a metaclass. We leave this here for backwards compatibility for now
# in case someone has been using the ManagedProperties class in downstream
# code. The reason that it might have been used is that when subclassing a
# class and wanting to use a metaclass the metaclass must be a subclass of the
# metaclass for the class that is being subclassed. Anyone wanting to subclass
# Basic and use a metaclass in their subclass would have needed to subclass
# ManagedProperties. Here ManagedProperties is not the metaclass for Basic any
# more but it should still be usable as a metaclass for Basic subclasses since
# it is a subclass of type which is now the metaclass for Basic.
class ManagedProperties(type):
    def __init__(cls, *args, **kwargs):
        msg = ("The ManagedProperties metaclass. "
               "Basic does not use metaclasses any more")
        sympy_deprecation_warning(msg,
            deprecated_since_version="1.12",
            active_deprecations_target='managedproperties')

        # Here we still call this function in case someone is using
        # ManagedProperties for something that is not a Basic subclass. For
        # Basic subclasses this function is now called by __init_subclass__ and
        # so this metaclass is not needed any more.
        _prepare_class_assumptions(cls)
