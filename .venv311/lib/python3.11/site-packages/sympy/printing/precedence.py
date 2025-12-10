"""A module providing information about the necessity of brackets"""


# Default precedence values for some basic types
PRECEDENCE = {
    "Lambda": 1,
    "Xor": 10,
    "Or": 20,
    "And": 30,
    "Relational": 35,
    "Add": 40,
    "Mul": 50,
    "Pow": 60,
    "Func": 70,
    "Not": 100,
    "Atom": 1000,
    "BitwiseOr": 36,
    "BitwiseXor": 37,
    "BitwiseAnd": 38
}

# A dictionary assigning precedence values to certain classes. These values are
# treated like they were inherited, so not every single class has to be named
# here.
# Do not use this with printers other than StrPrinter
PRECEDENCE_VALUES = {
    "Equivalent": PRECEDENCE["Xor"],
    "Xor": PRECEDENCE["Xor"],
    "Implies": PRECEDENCE["Xor"],
    "Or": PRECEDENCE["Or"],
    "And": PRECEDENCE["And"],
    "Add": PRECEDENCE["Add"],
    "Pow": PRECEDENCE["Pow"],
    "Relational": PRECEDENCE["Relational"],
    "Sub": PRECEDENCE["Add"],
    "Not": PRECEDENCE["Not"],
    "Function" : PRECEDENCE["Func"],
    "NegativeInfinity": PRECEDENCE["Add"],
    "MatAdd": PRECEDENCE["Add"],
    "MatPow": PRECEDENCE["Pow"],
    "MatrixSolve": PRECEDENCE["Mul"],
    "Mod": PRECEDENCE["Mul"],
    "TensAdd": PRECEDENCE["Add"],
    # As soon as `TensMul` is a subclass of `Mul`, remove this:
    "TensMul": PRECEDENCE["Mul"],
    "HadamardProduct": PRECEDENCE["Mul"],
    "HadamardPower": PRECEDENCE["Pow"],
    "KroneckerProduct": PRECEDENCE["Mul"],
    "Equality": PRECEDENCE["Mul"],
    "Unequality": PRECEDENCE["Mul"],
}

# Sometimes it's not enough to assign a fixed precedence value to a
# class. Then a function can be inserted in this dictionary that takes
# an instance of this class as argument and returns the appropriate
# precedence value.

# Precedence functions


def precedence_Mul(item):
    from sympy.core.function import Function
    if any(hasattr(arg, 'precedence') and isinstance(arg, Function) and
           arg.precedence < PRECEDENCE["Mul"] for arg in item.args):
        return PRECEDENCE["Mul"]

    if item.could_extract_minus_sign():
        return PRECEDENCE["Add"]
    return PRECEDENCE["Mul"]


def precedence_Rational(item):
    if item.p < 0:
        return PRECEDENCE["Add"]
    return PRECEDENCE["Mul"]


def precedence_Integer(item):
    if item.p < 0:
        return PRECEDENCE["Add"]
    return PRECEDENCE["Atom"]


def precedence_Float(item):
    if item < 0:
        return PRECEDENCE["Add"]
    return PRECEDENCE["Atom"]


def precedence_PolyElement(item):
    if item.is_generator:
        return PRECEDENCE["Atom"]
    elif item.is_ground:
        return precedence(item.coeff(1))
    elif item.is_term:
        return PRECEDENCE["Mul"]
    else:
        return PRECEDENCE["Add"]


def precedence_FracElement(item):
    if item.denom == 1:
        return precedence_PolyElement(item.numer)
    else:
        return PRECEDENCE["Mul"]


def precedence_UnevaluatedExpr(item):
    return precedence(item.args[0]) - 0.5


PRECEDENCE_FUNCTIONS = {
    "Integer": precedence_Integer,
    "Mul": precedence_Mul,
    "Rational": precedence_Rational,
    "Float": precedence_Float,
    "PolyElement": precedence_PolyElement,
    "FracElement": precedence_FracElement,
    "UnevaluatedExpr": precedence_UnevaluatedExpr,
}


def precedence(item):
    """Returns the precedence of a given object.

    This is the precedence for StrPrinter.
    """
    if hasattr(item, "precedence"):
        return item.precedence
    if not isinstance(item, type):
        for i in type(item).mro():
            n = i.__name__
            if n in PRECEDENCE_FUNCTIONS:
                return PRECEDENCE_FUNCTIONS[n](item)
            elif n in PRECEDENCE_VALUES:
                return PRECEDENCE_VALUES[n]
    return PRECEDENCE["Atom"]


PRECEDENCE_TRADITIONAL = PRECEDENCE.copy()
PRECEDENCE_TRADITIONAL['Integral'] = PRECEDENCE["Mul"]
PRECEDENCE_TRADITIONAL['Sum'] = PRECEDENCE["Mul"]
PRECEDENCE_TRADITIONAL['Product'] = PRECEDENCE["Mul"]
PRECEDENCE_TRADITIONAL['Limit'] = PRECEDENCE["Mul"]
PRECEDENCE_TRADITIONAL['Derivative'] = PRECEDENCE["Mul"]
PRECEDENCE_TRADITIONAL['TensorProduct'] = PRECEDENCE["Mul"]
PRECEDENCE_TRADITIONAL['Transpose'] = PRECEDENCE["Pow"]
PRECEDENCE_TRADITIONAL['Adjoint'] = PRECEDENCE["Pow"]
PRECEDENCE_TRADITIONAL['Dot'] = PRECEDENCE["Mul"] - 1
PRECEDENCE_TRADITIONAL['Cross'] = PRECEDENCE["Mul"] - 1
PRECEDENCE_TRADITIONAL['Gradient'] = PRECEDENCE["Mul"] - 1
PRECEDENCE_TRADITIONAL['Divergence'] = PRECEDENCE["Mul"] - 1
PRECEDENCE_TRADITIONAL['Curl'] = PRECEDENCE["Mul"] - 1
PRECEDENCE_TRADITIONAL['Laplacian'] = PRECEDENCE["Mul"] - 1
PRECEDENCE_TRADITIONAL['Union'] = PRECEDENCE['Xor']
PRECEDENCE_TRADITIONAL['Intersection'] = PRECEDENCE['Xor']
PRECEDENCE_TRADITIONAL['Complement'] = PRECEDENCE['Xor']
PRECEDENCE_TRADITIONAL['SymmetricDifference'] = PRECEDENCE['Xor']
PRECEDENCE_TRADITIONAL['ProductSet'] = PRECEDENCE['Xor']
PRECEDENCE_TRADITIONAL['DotProduct'] = PRECEDENCE_TRADITIONAL['Dot']


def precedence_traditional(item):
    """Returns the precedence of a given object according to the
    traditional rules of mathematics.

    This is the precedence for the LaTeX and pretty printer.
    """
    # Integral, Sum, Product, Limit have the precedence of Mul in LaTeX,
    # the precedence of Atom for other printers:
    from sympy.core.expr import UnevaluatedExpr

    if isinstance(item, UnevaluatedExpr):
        return precedence_traditional(item.args[0])

    n = item.__class__.__name__
    if n in PRECEDENCE_TRADITIONAL:
        return PRECEDENCE_TRADITIONAL[n]

    return precedence(item)
