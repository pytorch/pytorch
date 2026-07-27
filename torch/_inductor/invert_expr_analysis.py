from dataclasses import dataclass, replace

import sympy

from torch._inductor.utils import _IntLike, argsort_sym
from torch.utils._sympy.functions import FloorDiv, ModularIndexing

from .virtualized import V


def static_eq(a: _IntLike, b: _IntLike) -> bool:
    return V.graph.sizevars.statically_known_equals(a, b)


@dataclass(frozen=True)
class Term:
    coefficient: _IntLike
    range: _IntLike | None  # None for unbounded
    original_expr: sympy.Expr
    reconstruction_multiplier: _IntLike  # The multiplier needed for reconstruction


def generate_inverse_formula(
    expr: sympy.Expr, var: sympy.Symbol, var_range: _IntLike | None = None
) -> sympy.Expr | None:
    """
     Analyze an expression to see if it matches a specific invertible pattern that we
     know how to reverse.

     We're looking for expressions that are sums of terms where each term extracts a
     distinct bounded range from the input variable, like:

         y = c₀*a₀ + c₁*a₁ + c₂*a₂ + ... + cₙ*aₙ

     where each aᵢ must be one of these specific patterns:
     - ModularIndexing(var, divisor, modulo)
     - FloorDiv(ModularIndexing(var, 1, modulo), divisor)
     - FloorDiv(var, divisor)
     - var (the variable itself)

     The key pattern we need is:
     - Coefficients are strictly decreasing: c₀ > c₁ > c₂ > ... > cₙ
     - Each coefficient matches the product of ranges of later terms (mixed-radix property)
     - Each term extracts a bounded range, creating non-overlapping "slots"

     If we find this pattern, we can generate the reconstruction transformation that
     decomposes the variable and rebuilds it using the correct multipliers.

     EXAMPLE:
     Input: 100*((p//100)) + 10*((p%100)//10) + (p%10)

     Returns the reconstruction expression:
         remainder₀ = p
         component₀ = remainder₀ // 100          # hundreds digit
         remainder₁ = remainder₀ % 100
         component₁ = remainder₁ // 10           # tens digit
         remainder₂ = remainder₁ % 10
         component₂ = remainder₂                 # ones digit
         result = component₀*100 + component₁*10 + component₂*1

    This decomposes p into its components and rebuilds it using the original
     multipliers, which should equal the input expression.

     Args:
         expr: Expression to analyze (sum of terms with ModularIndexing, FloorDiv, etc.)
         var: The variable being decomposed
         var_range: Optional exclusive upper bound for nonnegative var

     Returns:
         None if not invertible, or the reconstruction expression

     References:
         Mixed-radix systems: https://en.wikipedia.org/wiki/Mixed_radix
    """
    # Step 1: Parse all terms
    terms = parse_terms(expr, var)
    if not terms:
        return None

    # Step 2: Sort by coefficient (descending)
    coeffs = [t.coefficient for t in terms]
    idxs = reversed(argsort_sym(V.graph.sizevars.shape_env, coeffs))
    terms = [terms[i] for i in idxs]

    # Step 3: Check invertibility conditions
    if not check_invertibility(terms, var_range):
        return None

    reconstruction = generate_reconstruction_expr(terms, var)
    if var_range is not None:
        reconstruction = V.graph.sizevars.simplify_with_ranges(
            reconstruction, {var: var_range}
        )
    return reconstruction


def parse_terms(expr: sympy.Expr, var: sympy.Symbol) -> list[Term] | None:
    """Parse expression into terms."""
    if not isinstance(expr, sympy.Add):
        # Single term
        term = parse_single_term(expr, var)
        return [term] if term else []

    terms = []
    for arg in expr.args:
        term = parse_single_term(arg, var)
        if term:
            terms.append(term)
        else:
            return None  # If any term fails to parse, fail completely

    return terms


def parse_single_term(term: sympy.Expr, var: sympy.Symbol) -> Term | None:
    """Parse a single term and extract coefficient, range, and reconstruction multiplier."""
    # Extract coefficient and expression parts
    coefficient, expr_parts = term.as_coeff_mul()

    if len(expr_parts) == 0:
        return None
    elif len(expr_parts) == 1:
        expr = expr_parts[0]
    else:
        # Multiple non-constant factors, too complex
        return None

    # Now determine the range and reconstruction multiplier
    range_val, reconstruction_multiplier = analyze_expression_properties(expr, var)
    if reconstruction_multiplier is None:
        return None

    return Term(
        coefficient=coefficient,
        range=range_val,
        original_expr=expr,
        reconstruction_multiplier=reconstruction_multiplier,
    )


def analyze_expression_properties(
    expr: sympy.Expr, var: sympy.Symbol
) -> tuple[_IntLike | None, _IntLike | None]:
    """Analyze an expression to determine its range and reconstruction multiplier."""
    # ModularIndexing(var, divisor, modulo) = (var // divisor) % modulo
    if isinstance(expr, ModularIndexing):
        x, div, mod = expr.args
        if static_eq(x, var):
            return mod, div  # Range is mod, multiplier is div

    # FloorDiv cases
    if isinstance(expr, FloorDiv):
        base, divisor = expr.args

        # FloorDiv(ModularIndexing(var, 1, mod), div) = (var % mod) // div
        if isinstance(base, ModularIndexing):
            x, inner_div, mod = base.args
            if static_eq(x, var) and static_eq(inner_div, 1):
                if not V.graph.sizevars.statically_known_multiple_of(mod, divisor):
                    return None, None
                range_val = FloorDiv(mod, divisor)
                return range_val, divisor  # Range is mod//div, multiplier is div

        # FloorDiv(var, divisor) = var // divisor (unbounded)
        elif static_eq(base, var):
            return None, divisor  # Unbounded range, multiplier is div

    return None, None


def check_invertibility(terms: list[Term], var_range: _IntLike | None = None) -> bool:
    """Check if the terms represent an invertible transformation."""
    if not terms:
        return False

    effective_terms: list[Term] = []
    for term in terms:
        if term.range is None and var_range is not None:
            multiplier = term.reconstruction_multiplier
            if not V.graph.sizevars.statically_known_multiple_of(var_range, multiplier):
                return False
            term = replace(term, range=FloorDiv(var_range, multiplier))
        effective_terms.append(term)
    terms = effective_terms

    # Coefficients must be strictly decreasing
    coeffs = [t.coefficient for t in terms]
    if argsort_sym(V.graph.sizevars.shape_env, coeffs) != list(
        reversed(range(len(coeffs)))
    ):
        return False

    # Check mixed-radix property: each coeff[i] = coeff[i+1] * range[i+1]
    ascending_terms = list(reversed(terms))
    expected_coeff = 1
    for i, term in enumerate(ascending_terms):
        if not static_eq(term.coefficient, expected_coeff):
            return False
        if term.range is None:
            if i + 1 < len(ascending_terms):
                return False
            continue
        expected_coeff = term.coefficient * term.range

    return check_reconstruction_coverage(terms, var_range)


def check_reconstruction_coverage(
    terms: list[Term],
    var_range: _IntLike | None = None,
) -> bool:
    """Check that terms reconstruct non-overlapping contiguous source ranges."""
    if not terms:
        return False

    multipliers = [t.reconstruction_multiplier for t in terms]
    idxs = argsort_sym(V.graph.sizevars.shape_env, multipliers)
    ascending_terms = [terms[i] for i in idxs]

    expected_multiplier = 1
    for i, term in enumerate(ascending_terms):
        if not static_eq(term.reconstruction_multiplier, expected_multiplier):
            return False
        if term.range is None:
            return i == len(ascending_terms) - 1 and var_range is None
        expected_multiplier = term.reconstruction_multiplier * term.range

    if var_range is not None and not static_eq(var_range, expected_multiplier):
        return False

    return True


def generate_reconstruction_expr(terms: list[Term], var: sympy.Symbol) -> sympy.Expr:
    y = var
    reconstruction = sympy.S.Zero
    remainder = y

    for i, term in enumerate(terms):
        if i < len(terms) - 1:
            component = FloorDiv(remainder, term.coefficient)
            remainder = ModularIndexing(remainder, 1, term.coefficient)
        else:
            # Last term should also divide by its coefficient
            component = FloorDiv(remainder, term.coefficient)

        reconstruction += component * term.reconstruction_multiplier

    return reconstruction
