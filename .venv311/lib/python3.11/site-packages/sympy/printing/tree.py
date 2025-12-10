def pprint_nodes(subtrees):
    """
    Prettyprints systems of nodes.

    Examples
    ========

    >>> from sympy.printing.tree import pprint_nodes
    >>> print(pprint_nodes(["a", "b1\\nb2", "c"]))
    +-a
    +-b1
    | b2
    +-c

    """
    def indent(s, type=1):
        x = s.split("\n")
        r = "+-%s\n" % x[0]
        for a in x[1:]:
            if a == "":
                continue
            if type == 1:
                r += "| %s\n" % a
            else:
                r += "  %s\n" % a
        return r
    if not subtrees:
        return ""
    f = ""
    for a in subtrees[:-1]:
        f += indent(a)
    f += indent(subtrees[-1], 2)
    return f


def print_node(node, assumptions=True):
    """
    Returns information about the "node".

    This includes class name, string representation and assumptions.

    Parameters
    ==========

    assumptions : bool, optional
        See the ``assumptions`` keyword in ``tree``
    """
    s = "%s: %s\n" % (node.__class__.__name__, str(node))

    if assumptions:
        d = node._assumptions
    else:
        d = None

    if d:
        for a in sorted(d):
            v = d[a]
            if v is None:
                continue
            s += "%s: %s\n" % (a, v)

    return s


def tree(node, assumptions=True):
    """
    Returns a tree representation of "node" as a string.

    It uses print_node() together with pprint_nodes() on node.args recursively.

    Parameters
    ==========

    assumptions : bool, optional
        The flag to decide whether to print out all the assumption data
        (such as ``is_integer`, ``is_real``) associated with the
        expression or not.

        Enabling the flag makes the result verbose, and the printed
        result may not be deterministic because of the randomness used
        in backtracing the assumptions.

    See Also
    ========

    print_tree

    """
    subtrees = []
    for arg in node.args:
        subtrees.append(tree(arg, assumptions=assumptions))
    s = print_node(node, assumptions=assumptions) + pprint_nodes(subtrees)
    return s


def print_tree(node, assumptions=True):
    """
    Prints a tree representation of "node".

    Parameters
    ==========

    assumptions : bool, optional
        The flag to decide whether to print out all the assumption data
        (such as ``is_integer`, ``is_real``) associated with the
        expression or not.

        Enabling the flag makes the result verbose, and the printed
        result may not be deterministic because of the randomness used
        in backtracing the assumptions.

    Examples
    ========

    >>> from sympy.printing import print_tree
    >>> from sympy import Symbol
    >>> x = Symbol('x', odd=True)
    >>> y = Symbol('y', even=True)

    Printing with full assumptions information:

    >>> print_tree(y**x)
    Pow: y**x
    +-Symbol: y
    | algebraic: True
    | commutative: True
    | complex: True
    | even: True
    | extended_real: True
    | finite: True
    | hermitian: True
    | imaginary: False
    | infinite: False
    | integer: True
    | irrational: False
    | noninteger: False
    | odd: False
    | rational: True
    | real: True
    | transcendental: False
    +-Symbol: x
      algebraic: True
      commutative: True
      complex: True
      even: False
      extended_nonzero: True
      extended_real: True
      finite: True
      hermitian: True
      imaginary: False
      infinite: False
      integer: True
      irrational: False
      noninteger: False
      nonzero: True
      odd: True
      rational: True
      real: True
      transcendental: False
      zero: False

    Hiding the assumptions:

    >>> print_tree(y**x, assumptions=False)
    Pow: y**x
    +-Symbol: y
    +-Symbol: x

    See Also
    ========

    tree

    """
    print(tree(node, assumptions=assumptions))
