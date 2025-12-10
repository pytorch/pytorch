""" A module for mapping operators to their corresponding eigenstates
and vice versa

It contains a global dictionary with eigenstate-operator pairings.
If a new state-operator pair is created, this dictionary should be
updated as well.

It also contains functions operators_to_state and state_to_operators
for mapping between the two. These can handle both classes and
instances of operators and states. See the individual function
descriptions for details.

TODO List:
- Update the dictionary with a complete list of state-operator pairs
"""

from sympy.physics.quantum.cartesian import (XOp, YOp, ZOp, XKet, PxOp, PxKet,
                                             PositionKet3D)
from sympy.physics.quantum.operator import Operator
from sympy.physics.quantum.state import StateBase, BraBase, Ket
from sympy.physics.quantum.spin import (JxOp, JyOp, JzOp, J2Op, JxKet, JyKet,
                                        JzKet)

__all__ = [
    'operators_to_state',
    'state_to_operators'
]

#state_mapping stores the mappings between states and their associated
#operators or tuples of operators. This should be updated when new
#classes are written! Entries are of the form PxKet : PxOp or
#something like 3DKet : (ROp, ThetaOp, PhiOp)

#frozenset is used so that the reverse mapping can be made
#(regular sets are not hashable because they are mutable
state_mapping = { JxKet: frozenset((J2Op, JxOp)),
                  JyKet: frozenset((J2Op, JyOp)),
                  JzKet: frozenset((J2Op, JzOp)),
                  Ket: Operator,
                  PositionKet3D: frozenset((XOp, YOp, ZOp)),
                  PxKet: PxOp,
                  XKet: XOp }

op_mapping = {v: k for k, v in state_mapping.items()}


def operators_to_state(operators, **options):
    """ Returns the eigenstate of the given operator or set of operators

    A global function for mapping operator classes to their associated
    states. It takes either an Operator or a set of operators and
    returns the state associated with these.

    This function can handle both instances of a given operator or
    just the class itself (i.e. both XOp() and XOp)

    There are multiple use cases to consider:

    1) A class or set of classes is passed: First, we try to
    instantiate default instances for these operators. If this fails,
    then the class is simply returned. If we succeed in instantiating
    default instances, then we try to call state._operators_to_state
    on the operator instances. If this fails, the class is returned.
    Otherwise, the instance returned by _operators_to_state is returned.

    2) An instance or set of instances is passed: In this case,
    state._operators_to_state is called on the instances passed. If
    this fails, a state class is returned. If the method returns an
    instance, that instance is returned.

    In both cases, if the operator class or set does not exist in the
    state_mapping dictionary, None is returned.

    Parameters
    ==========

    arg: Operator or set
         The class or instance of the operator or set of operators
         to be mapped to a state

    Examples
    ========

    >>> from sympy.physics.quantum.cartesian import XOp, PxOp
    >>> from sympy.physics.quantum.operatorset import operators_to_state
    >>> from sympy.physics.quantum.operator import Operator
    >>> operators_to_state(XOp)
    |x>
    >>> operators_to_state(XOp())
    |x>
    >>> operators_to_state(PxOp)
    |px>
    >>> operators_to_state(PxOp())
    |px>
    >>> operators_to_state(Operator)
    |psi>
    >>> operators_to_state(Operator())
    |psi>
    """

    if not (isinstance(operators, (Operator, set)) or issubclass(operators, Operator)):
        raise NotImplementedError("Argument is not an Operator or a set!")

    if isinstance(operators, set):
        for s in operators:
            if not (isinstance(s, Operator)
                   or issubclass(s, Operator)):
                raise NotImplementedError("Set is not all Operators!")

        ops = frozenset(operators)

        if ops in op_mapping:  # ops is a list of classes in this case
            #Try to get an object from default instances of the
            #operators...if this fails, return the class
            try:
                op_instances = [op() for op in ops]
                ret = _get_state(op_mapping[ops], set(op_instances), **options)
            except NotImplementedError:
                ret = op_mapping[ops]

            return ret
        else:
            tmp = [type(o) for o in ops]
            classes = frozenset(tmp)

            if classes in op_mapping:
                ret = _get_state(op_mapping[classes], ops, **options)
            else:
                ret = None

            return ret
    else:
        if operators in op_mapping:
            try:
                op_instance = operators()
                ret = _get_state(op_mapping[operators], op_instance, **options)
            except NotImplementedError:
                ret = op_mapping[operators]

            return ret
        elif type(operators) in op_mapping:
            return _get_state(op_mapping[type(operators)], operators, **options)
        else:
            return None


def state_to_operators(state, **options):
    """ Returns the operator or set of operators corresponding to the
    given eigenstate

    A global function for mapping state classes to their associated
    operators or sets of operators. It takes either a state class
    or instance.

    This function can handle both instances of a given state or just
    the class itself (i.e. both XKet() and XKet)

    There are multiple use cases to consider:

    1) A state class is passed: In this case, we first try
    instantiating a default instance of the class. If this succeeds,
    then we try to call state._state_to_operators on that instance.
    If the creation of the default instance or if the calling of
    _state_to_operators fails, then either an operator class or set of
    operator classes is returned. Otherwise, the appropriate
    operator instances are returned.

    2) A state instance is returned: Here, state._state_to_operators
    is called for the instance. If this fails, then a class or set of
    operator classes is returned. Otherwise, the instances are returned.

    In either case, if the state's class does not exist in
    state_mapping, None is returned.

    Parameters
    ==========

    arg: StateBase class or instance (or subclasses)
         The class or instance of the state to be mapped to an
         operator or set of operators

    Examples
    ========

    >>> from sympy.physics.quantum.cartesian import XKet, PxKet, XBra, PxBra
    >>> from sympy.physics.quantum.operatorset import state_to_operators
    >>> from sympy.physics.quantum.state import Ket, Bra
    >>> state_to_operators(XKet)
    X
    >>> state_to_operators(XKet())
    X
    >>> state_to_operators(PxKet)
    Px
    >>> state_to_operators(PxKet())
    Px
    >>> state_to_operators(PxBra)
    Px
    >>> state_to_operators(XBra)
    X
    >>> state_to_operators(Ket)
    O
    >>> state_to_operators(Bra)
    O
    """

    if not (isinstance(state, StateBase) or issubclass(state, StateBase)):
        raise NotImplementedError("Argument is not a state!")

    if state in state_mapping:  # state is a class
        state_inst = _make_default(state)
        try:
            ret = _get_ops(state_inst,
                           _make_set(state_mapping[state]), **options)
        except (NotImplementedError, TypeError):
            ret = state_mapping[state]
    elif type(state) in state_mapping:
        ret = _get_ops(state,
                       _make_set(state_mapping[type(state)]), **options)
    elif isinstance(state, BraBase) and state.dual_class() in state_mapping:
        ret = _get_ops(state,
                       _make_set(state_mapping[state.dual_class()]))
    elif issubclass(state, BraBase) and state.dual_class() in state_mapping:
        state_inst = _make_default(state)
        try:
            ret = _get_ops(state_inst,
                           _make_set(state_mapping[state.dual_class()]))
        except (NotImplementedError, TypeError):
            ret = state_mapping[state.dual_class()]
    else:
        ret = None

    return _make_set(ret)


def _make_default(expr):
    # XXX: Catching TypeError like this is a bad way of distinguishing between
    # classes and instances. The logic using this function should be rewritten
    # somehow.
    try:
        ret = expr()
    except TypeError:
        ret = expr

    return ret


def _get_state(state_class, ops, **options):
    # Try to get a state instance from the operator INSTANCES.
    # If this fails, get the class
    try:
        ret = state_class._operators_to_state(ops, **options)
    except NotImplementedError:
        ret = _make_default(state_class)

    return ret


def _get_ops(state_inst, op_classes, **options):
    # Try to get operator instances from the state INSTANCE.
    # If this fails, just return the classes
    try:
        ret = state_inst._state_to_operators(op_classes, **options)
    except NotImplementedError:
        if isinstance(op_classes, (set, tuple, frozenset)):
            ret = tuple(_make_default(x) for x in op_classes)
        else:
            ret = _make_default(op_classes)

    if isinstance(ret, set) and len(ret) == 1:
        return ret[0]

    return ret


def _make_set(ops):
    if isinstance(ops, (tuple, list, frozenset)):
        return set(ops)
    else:
        return ops
