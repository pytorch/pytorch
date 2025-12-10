"""Polynomial manipulation algorithms and algebraic objects. """

__all__ = [
    'Poly', 'PurePoly', 'poly_from_expr', 'parallel_poly_from_expr', 'degree',
    'total_degree', 'degree_list', 'LC', 'LM', 'LT', 'pdiv', 'prem', 'pquo',
    'pexquo', 'div', 'rem', 'quo', 'exquo', 'half_gcdex', 'gcdex', 'invert',
    'subresultants', 'resultant', 'discriminant', 'cofactors', 'gcd_list',
    'gcd', 'lcm_list', 'lcm', 'terms_gcd', 'trunc', 'monic', 'content',
    'primitive', 'compose', 'decompose', 'sturm', 'gff_list', 'gff',
    'sqf_norm', 'sqf_part', 'sqf_list', 'sqf', 'factor_list', 'factor',
    'intervals', 'refine_root', 'count_roots', 'all_roots', 'real_roots',
    'nroots', 'ground_roots', 'nth_power_roots_poly', 'cancel', 'reduced',
    'groebner', 'is_zero_dimensional', 'GroebnerBasis', 'poly',

    'symmetrize', 'horner', 'interpolate', 'rational_interpolate', 'viete',

    'together',

    'BasePolynomialError', 'ExactQuotientFailed', 'PolynomialDivisionFailed',
    'OperationNotSupported', 'HeuristicGCDFailed', 'HomomorphismFailed',
    'IsomorphismFailed', 'ExtraneousFactors', 'EvaluationFailed',
    'RefinementFailed', 'CoercionFailed', 'NotInvertible', 'NotReversible',
    'NotAlgebraic', 'DomainError', 'PolynomialError', 'UnificationFailed',
    'GeneratorsError', 'GeneratorsNeeded', 'ComputationFailed',
    'UnivariatePolynomialError', 'MultivariatePolynomialError',
    'PolificationFailed', 'OptionError', 'FlagError',

    'minpoly', 'minimal_polynomial', 'primitive_element', 'field_isomorphism',
    'to_number_field', 'isolate', 'round_two', 'prime_decomp',
    'prime_valuation', 'galois_group',

    'itermonomials', 'Monomial',

    'lex', 'grlex', 'grevlex', 'ilex', 'igrlex', 'igrevlex',

    'CRootOf', 'rootof', 'RootOf', 'ComplexRootOf', 'RootSum',

    'roots',

    'Domain', 'FiniteField', 'IntegerRing', 'RationalField', 'RealField',
    'ComplexField', 'PythonFiniteField', 'GMPYFiniteField',
    'PythonIntegerRing', 'GMPYIntegerRing', 'PythonRational',
    'GMPYRationalField', 'AlgebraicField', 'PolynomialRing', 'FractionField',
    'ExpressionDomain', 'FF_python', 'FF_gmpy', 'ZZ_python', 'ZZ_gmpy',
    'QQ_python', 'QQ_gmpy', 'GF', 'FF', 'ZZ', 'QQ', 'ZZ_I', 'QQ_I', 'RR',
    'CC', 'EX', 'EXRAW',

    'construct_domain',

    'swinnerton_dyer_poly', 'cyclotomic_poly', 'symmetric_poly',
    'random_poly', 'interpolating_poly',

    'jacobi_poly', 'chebyshevt_poly', 'chebyshevu_poly', 'hermite_poly',
    'hermite_prob_poly', 'legendre_poly', 'laguerre_poly',

    'bernoulli_poly', 'bernoulli_c_poly', 'genocchi_poly', 'euler_poly',
    'andre_poly',

    'apart', 'apart_list', 'assemble_partfrac_list',

    'Options',

    'ring', 'xring', 'vring', 'sring',

    'field', 'xfield', 'vfield', 'sfield'
]

from .polytools import (Poly, PurePoly, poly_from_expr,
        parallel_poly_from_expr, degree, total_degree, degree_list, LC, LM,
        LT, pdiv, prem, pquo, pexquo, div, rem, quo, exquo, half_gcdex, gcdex,
        invert, subresultants, resultant, discriminant, cofactors, gcd_list,
        gcd, lcm_list, lcm, terms_gcd, trunc, monic, content, primitive,
        compose, decompose, sturm, gff_list, gff, sqf_norm, sqf_part,
        sqf_list, sqf, factor_list, factor, intervals, refine_root,
        count_roots, all_roots, real_roots, nroots, ground_roots,
        nth_power_roots_poly, cancel, reduced, groebner, is_zero_dimensional,
        GroebnerBasis, poly)

from .polyfuncs import (symmetrize, horner, interpolate,
        rational_interpolate, viete)

from .rationaltools import together

from .polyerrors import (BasePolynomialError, ExactQuotientFailed,
        PolynomialDivisionFailed, OperationNotSupported, HeuristicGCDFailed,
        HomomorphismFailed, IsomorphismFailed, ExtraneousFactors,
        EvaluationFailed, RefinementFailed, CoercionFailed, NotInvertible,
        NotReversible, NotAlgebraic, DomainError, PolynomialError,
        UnificationFailed, GeneratorsError, GeneratorsNeeded,
        ComputationFailed, UnivariatePolynomialError,
        MultivariatePolynomialError, PolificationFailed, OptionError,
        FlagError)

from .numberfields import (minpoly, minimal_polynomial, primitive_element,
        field_isomorphism, to_number_field, isolate, round_two, prime_decomp,
        prime_valuation, galois_group)

from .monomials import itermonomials, Monomial

from .orderings import lex, grlex, grevlex, ilex, igrlex, igrevlex

from .rootoftools import CRootOf, rootof, RootOf, ComplexRootOf, RootSum

from .polyroots import roots

from .domains import (Domain, FiniteField, IntegerRing, RationalField,
        RealField, ComplexField, PythonFiniteField, GMPYFiniteField,
        PythonIntegerRing, GMPYIntegerRing, PythonRational, GMPYRationalField,
        AlgebraicField, PolynomialRing, FractionField, ExpressionDomain,
        FF_python, FF_gmpy, ZZ_python, ZZ_gmpy, QQ_python, QQ_gmpy, GF, FF,
        ZZ, QQ, ZZ_I, QQ_I, RR, CC, EX, EXRAW)

from .constructor import construct_domain

from .specialpolys import (swinnerton_dyer_poly, cyclotomic_poly,
        symmetric_poly, random_poly, interpolating_poly)

from .orthopolys import (jacobi_poly, chebyshevt_poly, chebyshevu_poly,
        hermite_poly, hermite_prob_poly, legendre_poly, laguerre_poly)

from .appellseqs import (bernoulli_poly, bernoulli_c_poly, genocchi_poly,
        euler_poly, andre_poly)

from .partfrac import apart, apart_list, assemble_partfrac_list

from .polyoptions import Options

from .rings import ring, xring, vring, sring

from .fields import field, xfield, vfield, sfield
