from sympy.printing.smtlib import smtlib_code
from sympy.assumptions.assume import AppliedPredicate
from sympy.assumptions.cnf import EncodedCNF
from sympy.assumptions.ask import Q

from sympy.core import Add, Mul
from sympy.core.relational import Equality, LessThan, GreaterThan, StrictLessThan, StrictGreaterThan
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import Pow
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.logic.boolalg import And, Or, Xor, Implies
from sympy.logic.boolalg import Not, ITE
from sympy.assumptions.relation.equality import StrictGreaterThanPredicate, StrictLessThanPredicate, GreaterThanPredicate, LessThanPredicate, EqualityPredicate
from sympy.external import import_module

def z3_satisfiable(expr, all_models=False):
    if not isinstance(expr, EncodedCNF):
        exprs = EncodedCNF()
        exprs.add_prop(expr)
        expr = exprs

    z3 = import_module("z3")
    if z3 is None:
        raise ImportError("z3 is not installed")

    s = encoded_cnf_to_z3_solver(expr, z3)

    res = str(s.check())
    if res == "unsat":
        return False
    elif res == "sat":
        return z3_model_to_sympy_model(s.model(), expr)
    else:
        return None


def z3_model_to_sympy_model(z3_model, enc_cnf):
    rev_enc = {value : key for key, value in enc_cnf.encoding.items()}
    return {rev_enc[int(var.name()[1:])] : bool(z3_model[var]) for var in z3_model}


def clause_to_assertion(clause):
    clause_strings = [f"d{abs(lit)}" if lit > 0 else f"(not d{abs(lit)})" for lit in clause]
    return "(assert (or " + " ".join(clause_strings) + "))"


def encoded_cnf_to_z3_solver(enc_cnf, z3):
    def dummify_bool(pred):
        return False
        assert isinstance(pred, AppliedPredicate)

        if pred.function in [Q.positive, Q.negative, Q.zero]:
            return pred
        else:
            return False

    s = z3.Solver()

    declarations = [f"(declare-const d{var} Bool)" for var in enc_cnf.variables]
    assertions = [clause_to_assertion(clause) for clause in enc_cnf.data]

    symbols = set()
    for pred, enc in enc_cnf.encoding.items():
        if not isinstance(pred, AppliedPredicate):
            continue
        if pred.function not in (Q.gt, Q.lt, Q.ge, Q.le, Q.ne, Q.eq, Q.positive, Q.negative, Q.extended_negative, Q.extended_positive, Q.zero, Q.nonzero, Q.nonnegative, Q.nonpositive, Q.extended_nonzero, Q.extended_nonnegative, Q.extended_nonpositive):
            continue

        pred_str = smtlib_code(pred, auto_declare=False, auto_assert=False, known_functions=known_functions)

        symbols |= pred.free_symbols
        pred = pred_str
        clause = f"(implies d{enc} {pred})"
        assertion = "(assert " + clause + ")"
        assertions.append(assertion)

    for sym in symbols:
        declarations.append(f"(declare-const {sym} Real)")

    declarations = "\n".join(declarations)
    assertions = "\n".join(assertions)
    s.from_string(declarations)
    s.from_string(assertions)

    return s


known_functions = {
            Add: '+',
            Mul: '*',

            Equality: '=',
            LessThan: '<=',
            GreaterThan: '>=',
            StrictLessThan: '<',
            StrictGreaterThan: '>',

            EqualityPredicate(): '=',
            LessThanPredicate(): '<=',
            GreaterThanPredicate(): '>=',
            StrictLessThanPredicate(): '<',
            StrictGreaterThanPredicate(): '>',

            Abs: 'abs',
            Min: 'min',
            Max: 'max',
            Pow: '^',

            And: 'and',
            Or: 'or',
            Xor: 'xor',
            Not: 'not',
            ITE: 'ite',
            Implies: '=>',
        }
