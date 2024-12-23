"""Compatibility interface between dense and sparse polys. """


from sympy.polys.densearith import dup_add_term
from sympy.polys.densearith import dmp_add_term
from sympy.polys.densearith import dup_sub_term
from sympy.polys.densearith import dmp_sub_term
from sympy.polys.densearith import dup_mul_term
from sympy.polys.densearith import dmp_mul_term
from sympy.polys.densearith import dup_add_ground
from sympy.polys.densearith import dmp_add_ground
from sympy.polys.densearith import dup_sub_ground
from sympy.polys.densearith import dmp_sub_ground
from sympy.polys.densearith import dup_mul_ground
from sympy.polys.densearith import dmp_mul_ground
from sympy.polys.densearith import dup_quo_ground
from sympy.polys.densearith import dmp_quo_ground
from sympy.polys.densearith import dup_exquo_ground
from sympy.polys.densearith import dmp_exquo_ground
from sympy.polys.densearith import dup_lshift
from sympy.polys.densearith import dup_rshift
from sympy.polys.densearith import dup_abs
from sympy.polys.densearith import dmp_abs
from sympy.polys.densearith import dup_neg
from sympy.polys.densearith import dmp_neg
from sympy.polys.densearith import dup_add
from sympy.polys.densearith import dmp_add
from sympy.polys.densearith import dup_sub
from sympy.polys.densearith import dmp_sub
from sympy.polys.densearith import dup_add_mul
from sympy.polys.densearith import dmp_add_mul
from sympy.polys.densearith import dup_sub_mul
from sympy.polys.densearith import dmp_sub_mul
from sympy.polys.densearith import dup_mul
from sympy.polys.densearith import dmp_mul
from sympy.polys.densearith import dup_sqr
from sympy.polys.densearith import dmp_sqr
from sympy.polys.densearith import dup_pow
from sympy.polys.densearith import dmp_pow
from sympy.polys.densearith import dup_pdiv
from sympy.polys.densearith import dup_prem
from sympy.polys.densearith import dup_pquo
from sympy.polys.densearith import dup_pexquo
from sympy.polys.densearith import dmp_pdiv
from sympy.polys.densearith import dmp_prem
from sympy.polys.densearith import dmp_pquo
from sympy.polys.densearith import dmp_pexquo
from sympy.polys.densearith import dup_rr_div
from sympy.polys.densearith import dmp_rr_div
from sympy.polys.densearith import dup_ff_div
from sympy.polys.densearith import dmp_ff_div
from sympy.polys.densearith import dup_div
from sympy.polys.densearith import dup_rem
from sympy.polys.densearith import dup_quo
from sympy.polys.densearith import dup_exquo
from sympy.polys.densearith import dmp_div
from sympy.polys.densearith import dmp_rem
from sympy.polys.densearith import dmp_quo
from sympy.polys.densearith import dmp_exquo
from sympy.polys.densearith import dup_max_norm
from sympy.polys.densearith import dmp_max_norm
from sympy.polys.densearith import dup_l1_norm
from sympy.polys.densearith import dmp_l1_norm
from sympy.polys.densearith import dup_l2_norm_squared
from sympy.polys.densearith import dmp_l2_norm_squared
from sympy.polys.densearith import dup_expand
from sympy.polys.densearith import dmp_expand
from sympy.polys.densebasic import dup_LC
from sympy.polys.densebasic import dmp_LC
from sympy.polys.densebasic import dup_TC
from sympy.polys.densebasic import dmp_TC
from sympy.polys.densebasic import dmp_ground_LC
from sympy.polys.densebasic import dmp_ground_TC
from sympy.polys.densebasic import dup_degree
from sympy.polys.densebasic import dmp_degree
from sympy.polys.densebasic import dmp_degree_in
from sympy.polys.densebasic import dmp_to_dict
from sympy.polys.densetools import dup_integrate
from sympy.polys.densetools import dmp_integrate
from sympy.polys.densetools import dmp_integrate_in
from sympy.polys.densetools import dup_diff
from sympy.polys.densetools import dmp_diff
from sympy.polys.densetools import dmp_diff_in
from sympy.polys.densetools import dup_eval
from sympy.polys.densetools import dmp_eval
from sympy.polys.densetools import dmp_eval_in
from sympy.polys.densetools import dmp_eval_tail
from sympy.polys.densetools import dmp_diff_eval_in
from sympy.polys.densetools import dup_trunc
from sympy.polys.densetools import dmp_trunc
from sympy.polys.densetools import dmp_ground_trunc
from sympy.polys.densetools import dup_monic
from sympy.polys.densetools import dmp_ground_monic
from sympy.polys.densetools import dup_content
from sympy.polys.densetools import dmp_ground_content
from sympy.polys.densetools import dup_primitive
from sympy.polys.densetools import dmp_ground_primitive
from sympy.polys.densetools import dup_extract
from sympy.polys.densetools import dmp_ground_extract
from sympy.polys.densetools import dup_real_imag
from sympy.polys.densetools import dup_mirror
from sympy.polys.densetools import dup_scale
from sympy.polys.densetools import dup_shift
from sympy.polys.densetools import dmp_shift
from sympy.polys.densetools import dup_transform
from sympy.polys.densetools import dup_compose
from sympy.polys.densetools import dmp_compose
from sympy.polys.densetools import dup_decompose
from sympy.polys.densetools import dmp_lift
from sympy.polys.densetools import dup_sign_variations
from sympy.polys.densetools import dup_clear_denoms
from sympy.polys.densetools import dmp_clear_denoms
from sympy.polys.densetools import dup_revert
from sympy.polys.euclidtools import dup_half_gcdex
from sympy.polys.euclidtools import dmp_half_gcdex
from sympy.polys.euclidtools import dup_gcdex
from sympy.polys.euclidtools import dmp_gcdex
from sympy.polys.euclidtools import dup_invert
from sympy.polys.euclidtools import dmp_invert
from sympy.polys.euclidtools import dup_euclidean_prs
from sympy.polys.euclidtools import dmp_euclidean_prs
from sympy.polys.euclidtools import dup_primitive_prs
from sympy.polys.euclidtools import dmp_primitive_prs
from sympy.polys.euclidtools import dup_inner_subresultants
from sympy.polys.euclidtools import dup_subresultants
from sympy.polys.euclidtools import dup_prs_resultant
from sympy.polys.euclidtools import dup_resultant
from sympy.polys.euclidtools import dmp_inner_subresultants
from sympy.polys.euclidtools import dmp_subresultants
from sympy.polys.euclidtools import dmp_prs_resultant
from sympy.polys.euclidtools import dmp_zz_modular_resultant
from sympy.polys.euclidtools import dmp_zz_collins_resultant
from sympy.polys.euclidtools import dmp_qq_collins_resultant
from sympy.polys.euclidtools import dmp_resultant
from sympy.polys.euclidtools import dup_discriminant
from sympy.polys.euclidtools import dmp_discriminant
from sympy.polys.euclidtools import dup_rr_prs_gcd
from sympy.polys.euclidtools import dup_ff_prs_gcd
from sympy.polys.euclidtools import dmp_rr_prs_gcd
from sympy.polys.euclidtools import dmp_ff_prs_gcd
from sympy.polys.euclidtools import dup_zz_heu_gcd
from sympy.polys.euclidtools import dmp_zz_heu_gcd
from sympy.polys.euclidtools import dup_qq_heu_gcd
from sympy.polys.euclidtools import dmp_qq_heu_gcd
from sympy.polys.euclidtools import dup_inner_gcd
from sympy.polys.euclidtools import dmp_inner_gcd
from sympy.polys.euclidtools import dup_gcd
from sympy.polys.euclidtools import dmp_gcd
from sympy.polys.euclidtools import dup_rr_lcm
from sympy.polys.euclidtools import dup_ff_lcm
from sympy.polys.euclidtools import dup_lcm
from sympy.polys.euclidtools import dmp_rr_lcm
from sympy.polys.euclidtools import dmp_ff_lcm
from sympy.polys.euclidtools import dmp_lcm
from sympy.polys.euclidtools import dmp_content
from sympy.polys.euclidtools import dmp_primitive
from sympy.polys.euclidtools import dup_cancel
from sympy.polys.euclidtools import dmp_cancel
from sympy.polys.factortools import dup_trial_division
from sympy.polys.factortools import dmp_trial_division
from sympy.polys.factortools import dup_zz_mignotte_bound
from sympy.polys.factortools import dmp_zz_mignotte_bound
from sympy.polys.factortools import dup_zz_hensel_step
from sympy.polys.factortools import dup_zz_hensel_lift
from sympy.polys.factortools import dup_zz_zassenhaus
from sympy.polys.factortools import dup_zz_irreducible_p
from sympy.polys.factortools import dup_cyclotomic_p
from sympy.polys.factortools import dup_zz_cyclotomic_poly
from sympy.polys.factortools import dup_zz_cyclotomic_factor
from sympy.polys.factortools import dup_zz_factor_sqf
from sympy.polys.factortools import dup_zz_factor
from sympy.polys.factortools import dmp_zz_wang_non_divisors
from sympy.polys.factortools import dmp_zz_wang_lead_coeffs
from sympy.polys.factortools import dup_zz_diophantine
from sympy.polys.factortools import dmp_zz_diophantine
from sympy.polys.factortools import dmp_zz_wang_hensel_lifting
from sympy.polys.factortools import dmp_zz_wang
from sympy.polys.factortools import dmp_zz_factor
from sympy.polys.factortools import dup_qq_i_factor
from sympy.polys.factortools import dup_zz_i_factor
from sympy.polys.factortools import dmp_qq_i_factor
from sympy.polys.factortools import dmp_zz_i_factor
from sympy.polys.factortools import dup_ext_factor
from sympy.polys.factortools import dmp_ext_factor
from sympy.polys.factortools import dup_gf_factor
from sympy.polys.factortools import dmp_gf_factor
from sympy.polys.factortools import dup_factor_list
from sympy.polys.factortools import dup_factor_list_include
from sympy.polys.factortools import dmp_factor_list
from sympy.polys.factortools import dmp_factor_list_include
from sympy.polys.factortools import dup_irreducible_p
from sympy.polys.factortools import dmp_irreducible_p
from sympy.polys.rootisolation import dup_sturm
from sympy.polys.rootisolation import dup_root_upper_bound
from sympy.polys.rootisolation import dup_root_lower_bound
from sympy.polys.rootisolation import dup_step_refine_real_root
from sympy.polys.rootisolation import dup_inner_refine_real_root
from sympy.polys.rootisolation import dup_outer_refine_real_root
from sympy.polys.rootisolation import dup_refine_real_root
from sympy.polys.rootisolation import dup_inner_isolate_real_roots
from sympy.polys.rootisolation import dup_inner_isolate_positive_roots
from sympy.polys.rootisolation import dup_inner_isolate_negative_roots
from sympy.polys.rootisolation import dup_isolate_real_roots_sqf
from sympy.polys.rootisolation import dup_isolate_real_roots
from sympy.polys.rootisolation import dup_isolate_real_roots_list
from sympy.polys.rootisolation import dup_count_real_roots
from sympy.polys.rootisolation import dup_count_complex_roots
from sympy.polys.rootisolation import dup_isolate_complex_roots_sqf
from sympy.polys.rootisolation import dup_isolate_all_roots_sqf
from sympy.polys.rootisolation import dup_isolate_all_roots

from sympy.polys.sqfreetools import (
    dup_sqf_p, dmp_sqf_p, dmp_norm, dup_sqf_norm, dmp_sqf_norm,
    dup_gf_sqf_part, dmp_gf_sqf_part, dup_sqf_part, dmp_sqf_part,
    dup_gf_sqf_list, dmp_gf_sqf_list, dup_sqf_list, dup_sqf_list_include,
    dmp_sqf_list, dmp_sqf_list_include, dup_gff_list, dmp_gff_list)

from sympy.polys.galoistools import (
    gf_degree, gf_LC, gf_TC, gf_strip, gf_from_dict,
    gf_to_dict, gf_from_int_poly, gf_to_int_poly, gf_neg, gf_add_ground, gf_sub_ground,
    gf_mul_ground, gf_quo_ground, gf_add, gf_sub, gf_mul, gf_sqr, gf_add_mul, gf_sub_mul,
    gf_expand, gf_div, gf_rem, gf_quo, gf_exquo, gf_lshift, gf_rshift, gf_pow, gf_pow_mod,
    gf_gcd, gf_lcm, gf_cofactors, gf_gcdex, gf_monic, gf_diff, gf_eval, gf_multi_eval,
    gf_compose, gf_compose_mod, gf_trace_map, gf_random, gf_irreducible, gf_irred_p_ben_or,
    gf_irred_p_rabin, gf_irreducible_p, gf_sqf_p, gf_sqf_part, gf_Qmatrix,
    gf_berlekamp, gf_ddf_zassenhaus, gf_edf_zassenhaus, gf_ddf_shoup, gf_edf_shoup,
    gf_zassenhaus, gf_shoup, gf_factor_sqf, gf_factor)

from sympy.utilities import public

@public
class IPolys:
    symbols = None
    ngens = None
    domain = None
    order = None
    gens = None

    def drop(self, gen):
        pass

    def clone(self, symbols=None, domain=None, order=None):
        pass

    def to_ground(self):
        pass

    def ground_new(self, element):
        pass

    def domain_new(self, element):
        pass

    def from_dict(self, d):
        pass

    def wrap(self, element):
        from sympy.polys.rings import PolyElement
        if isinstance(element, PolyElement):
            if element.ring == self:
                return element
            else:
                raise NotImplementedError("domain conversions")
        else:
            return self.ground_new(element)

    def to_dense(self, element):
        return self.wrap(element).to_dense()

    def from_dense(self, element):
        return self.from_dict(dmp_to_dict(element, self.ngens-1, self.domain))

    def dup_add_term(self, f, c, i):
        return self.from_dense(dup_add_term(self.to_dense(f), c, i, self.domain))
    def dmp_add_term(self, f, c, i):
        return self.from_dense(dmp_add_term(self.to_dense(f), self.wrap(c).drop(0).to_dense(), i, self.ngens-1, self.domain))
    def dup_sub_term(self, f, c, i):
        return self.from_dense(dup_sub_term(self.to_dense(f), c, i, self.domain))
    def dmp_sub_term(self, f, c, i):
        return self.from_dense(dmp_sub_term(self.to_dense(f), self.wrap(c).drop(0).to_dense(), i, self.ngens-1, self.domain))
    def dup_mul_term(self, f, c, i):
        return self.from_dense(dup_mul_term(self.to_dense(f), c, i, self.domain))
    def dmp_mul_term(self, f, c, i):
        return self.from_dense(dmp_mul_term(self.to_dense(f), self.wrap(c).drop(0).to_dense(), i, self.ngens-1, self.domain))

    def dup_add_ground(self, f, c):
        return self.from_dense(dup_add_ground(self.to_dense(f), c, self.domain))
    def dmp_add_ground(self, f, c):
        return self.from_dense(dmp_add_ground(self.to_dense(f), c, self.ngens-1, self.domain))
    def dup_sub_ground(self, f, c):
        return self.from_dense(dup_sub_ground(self.to_dense(f), c, self.domain))
    def dmp_sub_ground(self, f, c):
        return self.from_dense(dmp_sub_ground(self.to_dense(f), c, self.ngens-1, self.domain))
    def dup_mul_ground(self, f, c):
        return self.from_dense(dup_mul_ground(self.to_dense(f), c, self.domain))
    def dmp_mul_ground(self, f, c):
        return self.from_dense(dmp_mul_ground(self.to_dense(f), c, self.ngens-1, self.domain))
    def dup_quo_ground(self, f, c):
        return self.from_dense(dup_quo_ground(self.to_dense(f), c, self.domain))
    def dmp_quo_ground(self, f, c):
        return self.from_dense(dmp_quo_ground(self.to_dense(f), c, self.ngens-1, self.domain))
    def dup_exquo_ground(self, f, c):
        return self.from_dense(dup_exquo_ground(self.to_dense(f), c, self.domain))
    def dmp_exquo_ground(self, f, c):
        return self.from_dense(dmp_exquo_ground(self.to_dense(f), c, self.ngens-1, self.domain))

    def dup_lshift(self, f, n):
        return self.from_dense(dup_lshift(self.to_dense(f), n, self.domain))
    def dup_rshift(self, f, n):
        return self.from_dense(dup_rshift(self.to_dense(f), n, self.domain))

    def dup_abs(self, f):
        return self.from_dense(dup_abs(self.to_dense(f), self.domain))
    def dmp_abs(self, f):
        return self.from_dense(dmp_abs(self.to_dense(f), self.ngens-1, self.domain))

    def dup_neg(self, f):
        return self.from_dense(dup_neg(self.to_dense(f), self.domain))
    def dmp_neg(self, f):
        return self.from_dense(dmp_neg(self.to_dense(f), self.ngens-1, self.domain))

    def dup_add(self, f, g):
        return self.from_dense(dup_add(self.to_dense(f), self.to_dense(g), self.domain))
    def dmp_add(self, f, g):
        return self.from_dense(dmp_add(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))

    def dup_sub(self, f, g):
        return self.from_dense(dup_sub(self.to_dense(f), self.to_dense(g), self.domain))
    def dmp_sub(self, f, g):
        return self.from_dense(dmp_sub(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))

    def dup_add_mul(self, f, g, h):
        return self.from_dense(dup_add_mul(self.to_dense(f), self.to_dense(g), self.to_dense(h), self.domain))
    def dmp_add_mul(self, f, g, h):
        return self.from_dense(dmp_add_mul(self.to_dense(f), self.to_dense(g), self.to_dense(h), self.ngens-1, self.domain))
    def dup_sub_mul(self, f, g, h):
        return self.from_dense(dup_sub_mul(self.to_dense(f), self.to_dense(g), self.to_dense(h), self.domain))
    def dmp_sub_mul(self, f, g, h):
        return self.from_dense(dmp_sub_mul(self.to_dense(f), self.to_dense(g), self.to_dense(h), self.ngens-1, self.domain))

    def dup_mul(self, f, g):
        return self.from_dense(dup_mul(self.to_dense(f), self.to_dense(g), self.domain))
    def dmp_mul(self, f, g):
        return self.from_dense(dmp_mul(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))

    def dup_sqr(self, f):
        return self.from_dense(dup_sqr(self.to_dense(f), self.domain))
    def dmp_sqr(self, f):
        return self.from_dense(dmp_sqr(self.to_dense(f), self.ngens-1, self.domain))
    def dup_pow(self, f, n):
        return self.from_dense(dup_pow(self.to_dense(f), n, self.domain))
    def dmp_pow(self, f, n):
        return self.from_dense(dmp_pow(self.to_dense(f), n, self.ngens-1, self.domain))

    def dup_pdiv(self, f, g):
        q, r = dup_pdiv(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(q), self.from_dense(r))
    def dup_prem(self, f, g):
        return self.from_dense(dup_prem(self.to_dense(f), self.to_dense(g), self.domain))
    def dup_pquo(self, f, g):
        return self.from_dense(dup_pquo(self.to_dense(f), self.to_dense(g), self.domain))
    def dup_pexquo(self, f, g):
        return self.from_dense(dup_pexquo(self.to_dense(f), self.to_dense(g), self.domain))

    def dmp_pdiv(self, f, g):
        q, r = dmp_pdiv(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(q), self.from_dense(r))
    def dmp_prem(self, f, g):
        return self.from_dense(dmp_prem(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))
    def dmp_pquo(self, f, g):
        return self.from_dense(dmp_pquo(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))
    def dmp_pexquo(self, f, g):
        return self.from_dense(dmp_pexquo(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))

    def dup_rr_div(self, f, g):
        q, r = dup_rr_div(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(q), self.from_dense(r))
    def dmp_rr_div(self, f, g):
        q, r = dmp_rr_div(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(q), self.from_dense(r))
    def dup_ff_div(self, f, g):
        q, r = dup_ff_div(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(q), self.from_dense(r))
    def dmp_ff_div(self, f, g):
        q, r = dmp_ff_div(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(q), self.from_dense(r))

    def dup_div(self, f, g):
        q, r = dup_div(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(q), self.from_dense(r))
    def dup_rem(self, f, g):
        return self.from_dense(dup_rem(self.to_dense(f), self.to_dense(g), self.domain))
    def dup_quo(self, f, g):
        return self.from_dense(dup_quo(self.to_dense(f), self.to_dense(g), self.domain))
    def dup_exquo(self, f, g):
        return self.from_dense(dup_exquo(self.to_dense(f), self.to_dense(g), self.domain))

    def dmp_div(self, f, g):
        q, r = dmp_div(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(q), self.from_dense(r))
    def dmp_rem(self, f, g):
        return self.from_dense(dmp_rem(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))
    def dmp_quo(self, f, g):
        return self.from_dense(dmp_quo(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))
    def dmp_exquo(self, f, g):
        return self.from_dense(dmp_exquo(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))

    def dup_max_norm(self, f):
        return dup_max_norm(self.to_dense(f), self.domain)
    def dmp_max_norm(self, f):
        return dmp_max_norm(self.to_dense(f), self.ngens-1, self.domain)

    def dup_l1_norm(self, f):
        return dup_l1_norm(self.to_dense(f), self.domain)
    def dmp_l1_norm(self, f):
        return dmp_l1_norm(self.to_dense(f), self.ngens-1, self.domain)

    def dup_l2_norm_squared(self, f):
        return dup_l2_norm_squared(self.to_dense(f), self.domain)
    def dmp_l2_norm_squared(self, f):
        return dmp_l2_norm_squared(self.to_dense(f), self.ngens-1, self.domain)

    def dup_expand(self, polys):
        return self.from_dense(dup_expand(list(map(self.to_dense, polys)), self.domain))
    def dmp_expand(self, polys):
        return self.from_dense(dmp_expand(list(map(self.to_dense, polys)), self.ngens-1, self.domain))

    def dup_LC(self, f):
        return dup_LC(self.to_dense(f), self.domain)
    def dmp_LC(self, f):
        LC = dmp_LC(self.to_dense(f), self.domain)
        if isinstance(LC, list):
            return self[1:].from_dense(LC)
        else:
            return LC
    def dup_TC(self, f):
        return dup_TC(self.to_dense(f), self.domain)
    def dmp_TC(self, f):
        TC = dmp_TC(self.to_dense(f), self.domain)
        if isinstance(TC, list):
            return self[1:].from_dense(TC)
        else:
            return TC

    def dmp_ground_LC(self, f):
        return dmp_ground_LC(self.to_dense(f), self.ngens-1, self.domain)
    def dmp_ground_TC(self, f):
        return dmp_ground_TC(self.to_dense(f), self.ngens-1, self.domain)

    def dup_degree(self, f):
        return dup_degree(self.to_dense(f))
    def dmp_degree(self, f):
        return dmp_degree(self.to_dense(f), self.ngens-1)
    def dmp_degree_in(self, f, j):
        return dmp_degree_in(self.to_dense(f), j, self.ngens-1)
    def dup_integrate(self, f, m):
        return self.from_dense(dup_integrate(self.to_dense(f), m, self.domain))
    def dmp_integrate(self, f, m):
        return self.from_dense(dmp_integrate(self.to_dense(f), m, self.ngens-1, self.domain))

    def dup_diff(self, f, m):
        return self.from_dense(dup_diff(self.to_dense(f), m, self.domain))
    def dmp_diff(self, f, m):
        return self.from_dense(dmp_diff(self.to_dense(f), m, self.ngens-1, self.domain))

    def dmp_diff_in(self, f, m, j):
        return self.from_dense(dmp_diff_in(self.to_dense(f), m, j, self.ngens-1, self.domain))
    def dmp_integrate_in(self, f, m, j):
        return self.from_dense(dmp_integrate_in(self.to_dense(f), m, j, self.ngens-1, self.domain))

    def dup_eval(self, f, a):
        return dup_eval(self.to_dense(f), a, self.domain)
    def dmp_eval(self, f, a):
        result = dmp_eval(self.to_dense(f), a, self.ngens-1, self.domain)
        return self[1:].from_dense(result)

    def dmp_eval_in(self, f, a, j):
        result = dmp_eval_in(self.to_dense(f), a, j, self.ngens-1, self.domain)
        return self.drop(j).from_dense(result)
    def dmp_diff_eval_in(self, f, m, a, j):
        result = dmp_diff_eval_in(self.to_dense(f), m, a, j, self.ngens-1, self.domain)
        return self.drop(j).from_dense(result)

    def dmp_eval_tail(self, f, A):
        result = dmp_eval_tail(self.to_dense(f), A, self.ngens-1, self.domain)
        if isinstance(result, list):
            return self[:-len(A)].from_dense(result)
        else:
            return result

    def dup_trunc(self, f, p):
        return self.from_dense(dup_trunc(self.to_dense(f), p, self.domain))
    def dmp_trunc(self, f, g):
        return self.from_dense(dmp_trunc(self.to_dense(f), self[1:].to_dense(g), self.ngens-1, self.domain))
    def dmp_ground_trunc(self, f, p):
        return self.from_dense(dmp_ground_trunc(self.to_dense(f), p, self.ngens-1, self.domain))

    def dup_monic(self, f):
        return self.from_dense(dup_monic(self.to_dense(f), self.domain))
    def dmp_ground_monic(self, f):
        return self.from_dense(dmp_ground_monic(self.to_dense(f), self.ngens-1, self.domain))

    def dup_extract(self, f, g):
        c, F, G = dup_extract(self.to_dense(f), self.to_dense(g), self.domain)
        return (c, self.from_dense(F), self.from_dense(G))
    def dmp_ground_extract(self, f, g):
        c, F, G = dmp_ground_extract(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (c, self.from_dense(F), self.from_dense(G))

    def dup_real_imag(self, f):
        p, q = dup_real_imag(self.wrap(f).drop(1).to_dense(), self.domain)
        return (self.from_dense(p), self.from_dense(q))

    def dup_mirror(self, f):
        return self.from_dense(dup_mirror(self.to_dense(f), self.domain))
    def dup_scale(self, f, a):
        return self.from_dense(dup_scale(self.to_dense(f), a, self.domain))
    def dup_shift(self, f, a):
        return self.from_dense(dup_shift(self.to_dense(f), a, self.domain))
    def dmp_shift(self, f, a):
        return self.from_dense(dmp_shift(self.to_dense(f), a, self.ngens-1, self.domain))
    def dup_transform(self, f, p, q):
        return self.from_dense(dup_transform(self.to_dense(f), self.to_dense(p), self.to_dense(q), self.domain))

    def dup_compose(self, f, g):
        return self.from_dense(dup_compose(self.to_dense(f), self.to_dense(g), self.domain))
    def dmp_compose(self, f, g):
        return self.from_dense(dmp_compose(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))

    def dup_decompose(self, f):
        components = dup_decompose(self.to_dense(f), self.domain)
        return list(map(self.from_dense, components))

    def dmp_lift(self, f):
        result = dmp_lift(self.to_dense(f), self.ngens-1, self.domain)
        return self.to_ground().from_dense(result)

    def dup_sign_variations(self, f):
        return dup_sign_variations(self.to_dense(f), self.domain)

    def dup_clear_denoms(self, f, convert=False):
        c, F = dup_clear_denoms(self.to_dense(f), self.domain, convert=convert)
        if convert:
            ring = self.clone(domain=self.domain.get_ring())
        else:
            ring = self
        return (c, ring.from_dense(F))
    def dmp_clear_denoms(self, f, convert=False):
        c, F = dmp_clear_denoms(self.to_dense(f), self.ngens-1, self.domain, convert=convert)
        if convert:
            ring = self.clone(domain=self.domain.get_ring())
        else:
            ring = self
        return (c, ring.from_dense(F))

    def dup_revert(self, f, n):
        return self.from_dense(dup_revert(self.to_dense(f), n, self.domain))

    def dup_half_gcdex(self, f, g):
        s, h = dup_half_gcdex(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(s), self.from_dense(h))
    def dmp_half_gcdex(self, f, g):
        s, h = dmp_half_gcdex(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(s), self.from_dense(h))
    def dup_gcdex(self, f, g):
        s, t, h = dup_gcdex(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(s), self.from_dense(t), self.from_dense(h))
    def dmp_gcdex(self, f, g):
        s, t, h = dmp_gcdex(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(s), self.from_dense(t), self.from_dense(h))

    def dup_invert(self, f, g):
        return self.from_dense(dup_invert(self.to_dense(f), self.to_dense(g), self.domain))
    def dmp_invert(self, f, g):
        return self.from_dense(dmp_invert(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))

    def dup_euclidean_prs(self, f, g):
        prs = dup_euclidean_prs(self.to_dense(f), self.to_dense(g), self.domain)
        return list(map(self.from_dense, prs))
    def dmp_euclidean_prs(self, f, g):
        prs = dmp_euclidean_prs(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return list(map(self.from_dense, prs))
    def dup_primitive_prs(self, f, g):
        prs = dup_primitive_prs(self.to_dense(f), self.to_dense(g), self.domain)
        return list(map(self.from_dense, prs))
    def dmp_primitive_prs(self, f, g):
        prs = dmp_primitive_prs(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return list(map(self.from_dense, prs))

    def dup_inner_subresultants(self, f, g):
        prs, sres = dup_inner_subresultants(self.to_dense(f), self.to_dense(g), self.domain)
        return (list(map(self.from_dense, prs)), sres)
    def dmp_inner_subresultants(self, f, g):
        prs, sres  = dmp_inner_subresultants(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (list(map(self.from_dense, prs)), sres)

    def dup_subresultants(self, f, g):
        prs = dup_subresultants(self.to_dense(f), self.to_dense(g), self.domain)
        return list(map(self.from_dense, prs))
    def dmp_subresultants(self, f, g):
        prs = dmp_subresultants(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return list(map(self.from_dense, prs))

    def dup_prs_resultant(self, f, g):
        res, prs = dup_prs_resultant(self.to_dense(f), self.to_dense(g), self.domain)
        return (res, list(map(self.from_dense, prs)))
    def dmp_prs_resultant(self, f, g):
        res, prs = dmp_prs_resultant(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self[1:].from_dense(res), list(map(self.from_dense, prs)))

    def dmp_zz_modular_resultant(self, f, g, p):
        res = dmp_zz_modular_resultant(self.to_dense(f), self.to_dense(g), self.domain_new(p), self.ngens-1, self.domain)
        return self[1:].from_dense(res)
    def dmp_zz_collins_resultant(self, f, g):
        res = dmp_zz_collins_resultant(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return self[1:].from_dense(res)
    def dmp_qq_collins_resultant(self, f, g):
        res = dmp_qq_collins_resultant(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return self[1:].from_dense(res)

    def dup_resultant(self, f, g): #, includePRS=False):
        return dup_resultant(self.to_dense(f), self.to_dense(g), self.domain) #, includePRS=includePRS)
    def dmp_resultant(self, f, g): #, includePRS=False):
        res = dmp_resultant(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain) #, includePRS=includePRS)
        if isinstance(res, list):
            return self[1:].from_dense(res)
        else:
            return res

    def dup_discriminant(self, f):
        return dup_discriminant(self.to_dense(f), self.domain)
    def dmp_discriminant(self, f):
        disc = dmp_discriminant(self.to_dense(f), self.ngens-1, self.domain)
        if isinstance(disc, list):
            return self[1:].from_dense(disc)
        else:
            return disc

    def dup_rr_prs_gcd(self, f, g):
        H, F, G = dup_rr_prs_gcd(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))
    def dup_ff_prs_gcd(self, f, g):
        H, F, G = dup_ff_prs_gcd(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))
    def dmp_rr_prs_gcd(self, f, g):
        H, F, G = dmp_rr_prs_gcd(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))
    def dmp_ff_prs_gcd(self, f, g):
        H, F, G = dmp_ff_prs_gcd(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))
    def dup_zz_heu_gcd(self, f, g):
        H, F, G = dup_zz_heu_gcd(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))
    def dmp_zz_heu_gcd(self, f, g):
        H, F, G = dmp_zz_heu_gcd(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))
    def dup_qq_heu_gcd(self, f, g):
        H, F, G = dup_qq_heu_gcd(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))
    def dmp_qq_heu_gcd(self, f, g):
        H, F, G = dmp_qq_heu_gcd(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))
    def dup_inner_gcd(self, f, g):
        H, F, G = dup_inner_gcd(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))
    def dmp_inner_gcd(self, f, g):
        H, F, G = dmp_inner_gcd(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))
    def dup_gcd(self, f, g):
        H = dup_gcd(self.to_dense(f), self.to_dense(g), self.domain)
        return self.from_dense(H)
    def dmp_gcd(self, f, g):
        H = dmp_gcd(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return self.from_dense(H)
    def dup_rr_lcm(self, f, g):
        H = dup_rr_lcm(self.to_dense(f), self.to_dense(g), self.domain)
        return self.from_dense(H)
    def dup_ff_lcm(self, f, g):
        H = dup_ff_lcm(self.to_dense(f), self.to_dense(g), self.domain)
        return self.from_dense(H)
    def dup_lcm(self, f, g):
        H = dup_lcm(self.to_dense(f), self.to_dense(g), self.domain)
        return self.from_dense(H)
    def dmp_rr_lcm(self, f, g):
        H = dmp_rr_lcm(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return self.from_dense(H)
    def dmp_ff_lcm(self, f, g):
        H = dmp_ff_lcm(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return self.from_dense(H)
    def dmp_lcm(self, f, g):
        H = dmp_lcm(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return self.from_dense(H)

    def dup_content(self, f):
        cont = dup_content(self.to_dense(f), self.domain)
        return cont
    def dup_primitive(self, f):
        cont, prim = dup_primitive(self.to_dense(f), self.domain)
        return cont, self.from_dense(prim)

    def dmp_content(self, f):
        cont = dmp_content(self.to_dense(f), self.ngens-1, self.domain)
        if isinstance(cont, list):
            return self[1:].from_dense(cont)
        else:
            return cont
    def dmp_primitive(self, f):
        cont, prim = dmp_primitive(self.to_dense(f), self.ngens-1, self.domain)
        if isinstance(cont, list):
            return (self[1:].from_dense(cont), self.from_dense(prim))
        else:
            return (cont, self.from_dense(prim))

    def dmp_ground_content(self, f):
        cont = dmp_ground_content(self.to_dense(f), self.ngens-1, self.domain)
        return cont
    def dmp_ground_primitive(self, f):
        cont, prim = dmp_ground_primitive(self.to_dense(f), self.ngens-1, self.domain)
        return (cont, self.from_dense(prim))

    def dup_cancel(self, f, g, include=True):
        result = dup_cancel(self.to_dense(f), self.to_dense(g), self.domain, include=include)
        if not include:
            cf, cg, F, G = result
            return (cf, cg, self.from_dense(F), self.from_dense(G))
        else:
            F, G = result
            return (self.from_dense(F), self.from_dense(G))
    def dmp_cancel(self, f, g, include=True):
        result = dmp_cancel(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain, include=include)
        if not include:
            cf, cg, F, G = result
            return (cf, cg, self.from_dense(F), self.from_dense(G))
        else:
            F, G = result
            return (self.from_dense(F), self.from_dense(G))

    def dup_trial_division(self, f, factors):
        factors = dup_trial_division(self.to_dense(f), list(map(self.to_dense, factors)), self.domain)
        return [ (self.from_dense(g), k) for g, k in factors ]
    def dmp_trial_division(self, f, factors):
        factors = dmp_trial_division(self.to_dense(f), list(map(self.to_dense, factors)), self.ngens-1, self.domain)
        return [ (self.from_dense(g), k) for g, k in factors ]

    def dup_zz_mignotte_bound(self, f):
        return dup_zz_mignotte_bound(self.to_dense(f), self.domain)
    def dmp_zz_mignotte_bound(self, f):
        return dmp_zz_mignotte_bound(self.to_dense(f), self.ngens-1, self.domain)

    def dup_zz_hensel_step(self, m, f, g, h, s, t):
        D = self.to_dense
        G, H, S, T = dup_zz_hensel_step(m, D(f), D(g), D(h), D(s), D(t), self.domain)
        return (self.from_dense(G), self.from_dense(H), self.from_dense(S), self.from_dense(T))
    def dup_zz_hensel_lift(self, p, f, f_list, l):
        D = self.to_dense
        polys = dup_zz_hensel_lift(p, D(f), list(map(D, f_list)), l, self.domain)
        return list(map(self.from_dense, polys))

    def dup_zz_zassenhaus(self, f):
        factors = dup_zz_zassenhaus(self.to_dense(f), self.domain)
        return [ (self.from_dense(g), k) for g, k in factors ]

    def dup_zz_irreducible_p(self, f):
        return dup_zz_irreducible_p(self.to_dense(f), self.domain)
    def dup_cyclotomic_p(self, f, irreducible=False):
        return dup_cyclotomic_p(self.to_dense(f), self.domain, irreducible=irreducible)
    def dup_zz_cyclotomic_poly(self, n):
        F = dup_zz_cyclotomic_poly(n, self.domain)
        return self.from_dense(F)
    def dup_zz_cyclotomic_factor(self, f):
        result = dup_zz_cyclotomic_factor(self.to_dense(f), self.domain)
        if result is None:
            return result
        else:
            return list(map(self.from_dense, result))

    # E: List[ZZ], cs: ZZ, ct: ZZ
    def dmp_zz_wang_non_divisors(self, E, cs, ct):
        return dmp_zz_wang_non_divisors(E, cs, ct, self.domain)

    # f: Poly, T: List[(Poly, int)], ct: ZZ, A: List[ZZ]
    #def dmp_zz_wang_test_points(f, T, ct, A):
    #   dmp_zz_wang_test_points(self.to_dense(f), T, ct, A, self.ngens-1, self.domain)

    # f: Poly, T: List[(Poly, int)], cs: ZZ, E: List[ZZ], H: List[Poly], A: List[ZZ]
    def dmp_zz_wang_lead_coeffs(self, f, T, cs, E, H, A):
        mv = self[1:]
        T = [ (mv.to_dense(t), k) for t, k in T ]
        uv = self[:1]
        H = list(map(uv.to_dense, H))
        f, HH, CC = dmp_zz_wang_lead_coeffs(self.to_dense(f), T, cs, E, H, A, self.ngens-1, self.domain)
        return self.from_dense(f), list(map(uv.from_dense, HH)), list(map(mv.from_dense, CC))

    # f: List[Poly], m: int, p: ZZ
    def dup_zz_diophantine(self, F, m, p):
        result = dup_zz_diophantine(list(map(self.to_dense, F)), m, p, self.domain)
        return list(map(self.from_dense, result))

    # f: List[Poly], c: List[Poly], A: List[ZZ], d: int, p: ZZ
    def dmp_zz_diophantine(self, F, c, A, d, p):
        result = dmp_zz_diophantine(list(map(self.to_dense, F)), self.to_dense(c), A, d, p, self.ngens-1, self.domain)
        return list(map(self.from_dense, result))

    # f: Poly, H: List[Poly], LC: List[Poly], A: List[ZZ], p: ZZ
    def dmp_zz_wang_hensel_lifting(self, f, H, LC, A, p):
        uv = self[:1]
        mv = self[1:]
        H = list(map(uv.to_dense, H))
        LC = list(map(mv.to_dense, LC))
        result = dmp_zz_wang_hensel_lifting(self.to_dense(f), H, LC, A, p, self.ngens-1, self.domain)
        return list(map(self.from_dense, result))

    def dmp_zz_wang(self, f, mod=None, seed=None):
        factors = dmp_zz_wang(self.to_dense(f), self.ngens-1, self.domain, mod=mod, seed=seed)
        return [ self.from_dense(g) for g in factors ]

    def dup_zz_factor_sqf(self, f):
        coeff, factors = dup_zz_factor_sqf(self.to_dense(f), self.domain)
        return (coeff, [ self.from_dense(g) for g in factors ])

    def dup_zz_factor(self, f):
        coeff, factors = dup_zz_factor(self.to_dense(f), self.domain)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])
    def dmp_zz_factor(self, f):
        coeff, factors = dmp_zz_factor(self.to_dense(f), self.ngens-1, self.domain)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])

    def dup_qq_i_factor(self, f):
        coeff, factors = dup_qq_i_factor(self.to_dense(f), self.domain)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])
    def dmp_qq_i_factor(self, f):
        coeff, factors = dmp_qq_i_factor(self.to_dense(f), self.ngens-1, self.domain)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])

    def dup_zz_i_factor(self, f):
        coeff, factors = dup_zz_i_factor(self.to_dense(f), self.domain)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])
    def dmp_zz_i_factor(self, f):
        coeff, factors = dmp_zz_i_factor(self.to_dense(f), self.ngens-1, self.domain)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])

    def dup_ext_factor(self, f):
        coeff, factors = dup_ext_factor(self.to_dense(f), self.domain)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])
    def dmp_ext_factor(self, f):
        coeff, factors = dmp_ext_factor(self.to_dense(f), self.ngens-1, self.domain)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])

    def dup_gf_factor(self, f):
        coeff, factors = dup_gf_factor(self.to_dense(f), self.domain)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])
    def dmp_gf_factor(self, f):
        coeff, factors = dmp_gf_factor(self.to_dense(f), self.ngens-1, self.domain)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])

    def dup_factor_list(self, f):
        coeff, factors = dup_factor_list(self.to_dense(f), self.domain)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])
    def dup_factor_list_include(self, f):
        factors = dup_factor_list_include(self.to_dense(f), self.domain)
        return [ (self.from_dense(g), k) for g, k in factors ]

    def dmp_factor_list(self, f):
        coeff, factors = dmp_factor_list(self.to_dense(f), self.ngens-1, self.domain)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])
    def dmp_factor_list_include(self, f):
        factors = dmp_factor_list_include(self.to_dense(f), self.ngens-1, self.domain)
        return [ (self.from_dense(g), k) for g, k in factors ]

    def dup_irreducible_p(self, f):
        return dup_irreducible_p(self.to_dense(f), self.domain)
    def dmp_irreducible_p(self, f):
        return dmp_irreducible_p(self.to_dense(f), self.ngens-1, self.domain)

    def dup_sturm(self, f):
        seq = dup_sturm(self.to_dense(f), self.domain)
        return list(map(self.from_dense, seq))

    def dup_sqf_p(self, f):
        return dup_sqf_p(self.to_dense(f), self.domain)
    def dmp_sqf_p(self, f):
        return dmp_sqf_p(self.to_dense(f), self.ngens-1, self.domain)

    def dmp_norm(self, f):
        n = dmp_norm(self.to_dense(f), self.ngens-1, self.domain)
        return self.to_ground().from_dense(n)

    def dup_sqf_norm(self, f):
        s, F, R = dup_sqf_norm(self.to_dense(f), self.domain)
        return (s, self.from_dense(F), self.to_ground().from_dense(R))
    def dmp_sqf_norm(self, f):
        s, F, R = dmp_sqf_norm(self.to_dense(f), self.ngens-1, self.domain)
        return (s, self.from_dense(F), self.to_ground().from_dense(R))

    def dup_gf_sqf_part(self, f):
        return self.from_dense(dup_gf_sqf_part(self.to_dense(f), self.domain))
    def dmp_gf_sqf_part(self, f):
        return self.from_dense(dmp_gf_sqf_part(self.to_dense(f), self.domain))
    def dup_sqf_part(self, f):
        return self.from_dense(dup_sqf_part(self.to_dense(f), self.domain))
    def dmp_sqf_part(self, f):
        return self.from_dense(dmp_sqf_part(self.to_dense(f), self.ngens-1, self.domain))

    def dup_gf_sqf_list(self, f, all=False):
        coeff, factors = dup_gf_sqf_list(self.to_dense(f), self.domain, all=all)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])
    def dmp_gf_sqf_list(self, f, all=False):
        coeff, factors = dmp_gf_sqf_list(self.to_dense(f), self.ngens-1, self.domain, all=all)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])

    def dup_sqf_list(self, f, all=False):
        coeff, factors = dup_sqf_list(self.to_dense(f), self.domain, all=all)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])
    def dup_sqf_list_include(self, f, all=False):
        factors = dup_sqf_list_include(self.to_dense(f), self.domain, all=all)
        return [ (self.from_dense(g), k) for g, k in factors ]
    def dmp_sqf_list(self, f, all=False):
        coeff, factors = dmp_sqf_list(self.to_dense(f), self.ngens-1, self.domain, all=all)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])
    def dmp_sqf_list_include(self, f, all=False):
        factors = dmp_sqf_list_include(self.to_dense(f), self.ngens-1, self.domain, all=all)
        return [ (self.from_dense(g), k) for g, k in factors ]

    def dup_gff_list(self, f):
        factors = dup_gff_list(self.to_dense(f), self.domain)
        return [ (self.from_dense(g), k) for g, k in factors ]
    def dmp_gff_list(self, f):
        factors = dmp_gff_list(self.to_dense(f), self.ngens-1, self.domain)
        return [ (self.from_dense(g), k) for g, k in factors ]

    def dup_root_upper_bound(self, f):
        return dup_root_upper_bound(self.to_dense(f), self.domain)
    def dup_root_lower_bound(self, f):
        return dup_root_lower_bound(self.to_dense(f), self.domain)

    def dup_step_refine_real_root(self, f, M, fast=False):
        return dup_step_refine_real_root(self.to_dense(f), M, self.domain, fast=fast)
    def dup_inner_refine_real_root(self, f, M, eps=None, steps=None, disjoint=None, fast=False, mobius=False):
        return dup_inner_refine_real_root(self.to_dense(f), M, self.domain, eps=eps, steps=steps, disjoint=disjoint, fast=fast, mobius=mobius)
    def dup_outer_refine_real_root(self, f, s, t, eps=None, steps=None, disjoint=None, fast=False):
        return dup_outer_refine_real_root(self.to_dense(f), s, t, self.domain, eps=eps, steps=steps, disjoint=disjoint, fast=fast)
    def dup_refine_real_root(self, f, s, t, eps=None, steps=None, disjoint=None, fast=False):
        return dup_refine_real_root(self.to_dense(f), s, t, self.domain, eps=eps, steps=steps, disjoint=disjoint, fast=fast)
    def dup_inner_isolate_real_roots(self, f, eps=None, fast=False):
        return dup_inner_isolate_real_roots(self.to_dense(f), self.domain, eps=eps, fast=fast)
    def dup_inner_isolate_positive_roots(self, f, eps=None, inf=None, sup=None, fast=False, mobius=False):
        return dup_inner_isolate_positive_roots(self.to_dense(f), self.domain, eps=eps, inf=inf, sup=sup, fast=fast, mobius=mobius)
    def dup_inner_isolate_negative_roots(self, f, inf=None, sup=None, eps=None, fast=False, mobius=False):
        return dup_inner_isolate_negative_roots(self.to_dense(f), self.domain, inf=inf, sup=sup, eps=eps, fast=fast, mobius=mobius)
    def dup_isolate_real_roots_sqf(self, f, eps=None, inf=None, sup=None, fast=False, blackbox=False):
        return dup_isolate_real_roots_sqf(self.to_dense(f), self.domain, eps=eps, inf=inf, sup=sup, fast=fast, blackbox=blackbox)
    def dup_isolate_real_roots(self, f, eps=None, inf=None, sup=None, basis=False, fast=False):
        return dup_isolate_real_roots(self.to_dense(f), self.domain, eps=eps, inf=inf, sup=sup, basis=basis, fast=fast)
    def dup_isolate_real_roots_list(self, polys, eps=None, inf=None, sup=None, strict=False, basis=False, fast=False):
        return dup_isolate_real_roots_list(list(map(self.to_dense, polys)), self.domain, eps=eps, inf=inf, sup=sup, strict=strict, basis=basis, fast=fast)
    def dup_count_real_roots(self, f, inf=None, sup=None):
        return dup_count_real_roots(self.to_dense(f), self.domain, inf=inf, sup=sup)
    def dup_count_complex_roots(self, f, inf=None, sup=None, exclude=None):
        return dup_count_complex_roots(self.to_dense(f), self.domain, inf=inf, sup=sup, exclude=exclude)
    def dup_isolate_complex_roots_sqf(self, f, eps=None, inf=None, sup=None, blackbox=False):
        return dup_isolate_complex_roots_sqf(self.to_dense(f), self.domain, eps=eps, inf=inf, sup=sup, blackbox=blackbox)
    def dup_isolate_all_roots_sqf(self, f, eps=None, inf=None, sup=None, fast=False, blackbox=False):
        return dup_isolate_all_roots_sqf(self.to_dense(f), self.domain, eps=eps, inf=inf, sup=sup, fast=fast, blackbox=blackbox)
    def dup_isolate_all_roots(self, f, eps=None, inf=None, sup=None, fast=False):
        return dup_isolate_all_roots(self.to_dense(f), self.domain, eps=eps, inf=inf, sup=sup, fast=fast)

    def fateman_poly_F_1(self):
        from sympy.polys.specialpolys import dmp_fateman_poly_F_1
        return tuple(map(self.from_dense, dmp_fateman_poly_F_1(self.ngens-1, self.domain)))
    def fateman_poly_F_2(self):
        from sympy.polys.specialpolys import dmp_fateman_poly_F_2
        return tuple(map(self.from_dense, dmp_fateman_poly_F_2(self.ngens-1, self.domain)))
    def fateman_poly_F_3(self):
        from sympy.polys.specialpolys import dmp_fateman_poly_F_3
        return tuple(map(self.from_dense, dmp_fateman_poly_F_3(self.ngens-1, self.domain)))

    def to_gf_dense(self, element):
        return gf_strip([ self.domain.dom.convert(c, self.domain) for c in self.wrap(element).to_dense() ])

    def from_gf_dense(self, element):
        return self.from_dict(dmp_to_dict(element, self.ngens-1, self.domain.dom))

    def gf_degree(self, f):
        return gf_degree(self.to_gf_dense(f))

    def gf_LC(self, f):
        return gf_LC(self.to_gf_dense(f), self.domain.dom)
    def gf_TC(self, f):
        return gf_TC(self.to_gf_dense(f), self.domain.dom)

    def gf_strip(self, f):
        return self.from_gf_dense(gf_strip(self.to_gf_dense(f)))
    def gf_trunc(self, f):
        return self.from_gf_dense(gf_strip(self.to_gf_dense(f), self.domain.mod))
    def gf_normal(self, f):
        return self.from_gf_dense(gf_strip(self.to_gf_dense(f), self.domain.mod, self.domain.dom))

    def gf_from_dict(self, f):
        return self.from_gf_dense(gf_from_dict(f, self.domain.mod, self.domain.dom))
    def gf_to_dict(self, f, symmetric=True):
        return gf_to_dict(self.to_gf_dense(f), self.domain.mod, symmetric=symmetric)

    def gf_from_int_poly(self, f):
        return self.from_gf_dense(gf_from_int_poly(f, self.domain.mod))
    def gf_to_int_poly(self, f, symmetric=True):
        return gf_to_int_poly(self.to_gf_dense(f), self.domain.mod, symmetric=symmetric)

    def gf_neg(self, f):
        return self.from_gf_dense(gf_neg(self.to_gf_dense(f), self.domain.mod, self.domain.dom))

    def gf_add_ground(self, f, a):
        return self.from_gf_dense(gf_add_ground(self.to_gf_dense(f), a, self.domain.mod, self.domain.dom))
    def gf_sub_ground(self, f, a):
        return self.from_gf_dense(gf_sub_ground(self.to_gf_dense(f), a, self.domain.mod, self.domain.dom))
    def gf_mul_ground(self, f, a):
        return self.from_gf_dense(gf_mul_ground(self.to_gf_dense(f), a, self.domain.mod, self.domain.dom))
    def gf_quo_ground(self, f, a):
        return self.from_gf_dense(gf_quo_ground(self.to_gf_dense(f), a, self.domain.mod, self.domain.dom))

    def gf_add(self, f, g):
        return self.from_gf_dense(gf_add(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))
    def gf_sub(self, f, g):
        return self.from_gf_dense(gf_sub(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))
    def gf_mul(self, f, g):
        return self.from_gf_dense(gf_mul(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))
    def gf_sqr(self, f):
        return self.from_gf_dense(gf_sqr(self.to_gf_dense(f), self.domain.mod, self.domain.dom))

    def gf_add_mul(self, f, g, h):
        return self.from_gf_dense(gf_add_mul(self.to_gf_dense(f), self.to_gf_dense(g), self.to_gf_dense(h), self.domain.mod, self.domain.dom))
    def gf_sub_mul(self, f, g, h):
        return self.from_gf_dense(gf_sub_mul(self.to_gf_dense(f), self.to_gf_dense(g), self.to_gf_dense(h), self.domain.mod, self.domain.dom))

    def gf_expand(self, F):
        return self.from_gf_dense(gf_expand(list(map(self.to_gf_dense, F)), self.domain.mod, self.domain.dom))

    def gf_div(self, f, g):
        q, r = gf_div(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom)
        return self.from_gf_dense(q), self.from_gf_dense(r)
    def gf_rem(self, f, g):
        return self.from_gf_dense(gf_rem(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))
    def gf_quo(self, f, g):
        return self.from_gf_dense(gf_quo(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))
    def gf_exquo(self, f, g):
        return self.from_gf_dense(gf_exquo(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))

    def gf_lshift(self, f, n):
        return self.from_gf_dense(gf_lshift(self.to_gf_dense(f), n, self.domain.dom))
    def gf_rshift(self, f, n):
        return self.from_gf_dense(gf_rshift(self.to_gf_dense(f), n, self.domain.dom))

    def gf_pow(self, f, n):
        return self.from_gf_dense(gf_pow(self.to_gf_dense(f), n, self.domain.mod, self.domain.dom))
    def gf_pow_mod(self, f, n, g):
        return self.from_gf_dense(gf_pow_mod(self.to_gf_dense(f), n, self.to_gf_dense(g), self.domain.mod, self.domain.dom))

    def gf_cofactors(self, f, g):
        h, cff, cfg = gf_cofactors(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom)
        return self.from_gf_dense(h), self.from_gf_dense(cff), self.from_gf_dense(cfg)
    def gf_gcd(self, f, g):
        return self.from_gf_dense(gf_gcd(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))
    def gf_lcm(self, f, g):
        return self.from_gf_dense(gf_lcm(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))
    def gf_gcdex(self, f, g):
        return self.from_gf_dense(gf_gcdex(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))

    def gf_monic(self, f):
        return self.from_gf_dense(gf_monic(self.to_gf_dense(f), self.domain.mod, self.domain.dom))
    def gf_diff(self, f):
        return self.from_gf_dense(gf_diff(self.to_gf_dense(f), self.domain.mod, self.domain.dom))

    def gf_eval(self, f, a):
        return gf_eval(self.to_gf_dense(f), a, self.domain.mod, self.domain.dom)
    def gf_multi_eval(self, f, A):
        return gf_multi_eval(self.to_gf_dense(f), A, self.domain.mod, self.domain.dom)

    def gf_compose(self, f, g):
        return self.from_gf_dense(gf_compose(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))
    def gf_compose_mod(self, g, h, f):
        return self.from_gf_dense(gf_compose_mod(self.to_gf_dense(g), self.to_gf_dense(h), self.to_gf_dense(f), self.domain.mod, self.domain.dom))

    def gf_trace_map(self, a, b, c, n, f):
        a = self.to_gf_dense(a)
        b = self.to_gf_dense(b)
        c = self.to_gf_dense(c)
        f = self.to_gf_dense(f)
        U, V = gf_trace_map(a, b, c, n, f, self.domain.mod, self.domain.dom)
        return self.from_gf_dense(U), self.from_gf_dense(V)

    def gf_random(self, n):
        return self.from_gf_dense(gf_random(n, self.domain.mod, self.domain.dom))
    def gf_irreducible(self, n):
        return self.from_gf_dense(gf_irreducible(n, self.domain.mod, self.domain.dom))

    def gf_irred_p_ben_or(self, f):
        return gf_irred_p_ben_or(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
    def gf_irred_p_rabin(self, f):
        return gf_irred_p_rabin(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
    def gf_irreducible_p(self, f):
        return gf_irreducible_p(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
    def gf_sqf_p(self, f):
        return gf_sqf_p(self.to_gf_dense(f), self.domain.mod, self.domain.dom)

    def gf_sqf_part(self, f):
        return self.from_gf_dense(gf_sqf_part(self.to_gf_dense(f), self.domain.mod, self.domain.dom))
    def gf_sqf_list(self, f, all=False):
        coeff, factors = gf_sqf_part(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
        return coeff, [ (self.from_gf_dense(g), k) for g, k in factors ]

    def gf_Qmatrix(self, f):
        return gf_Qmatrix(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
    def gf_berlekamp(self, f):
        factors = gf_berlekamp(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
        return [ self.from_gf_dense(g) for g in factors ]

    def gf_ddf_zassenhaus(self, f):
        factors = gf_ddf_zassenhaus(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
        return [ (self.from_gf_dense(g), k) for g, k in factors ]
    def gf_edf_zassenhaus(self, f, n):
        factors = gf_edf_zassenhaus(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
        return [ self.from_gf_dense(g) for g in factors ]

    def gf_ddf_shoup(self, f):
        factors = gf_ddf_shoup(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
        return [ (self.from_gf_dense(g), k) for g, k in factors ]
    def gf_edf_shoup(self, f, n):
        factors = gf_edf_shoup(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
        return [ self.from_gf_dense(g) for g in factors ]

    def gf_zassenhaus(self, f):
        factors = gf_zassenhaus(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
        return [ self.from_gf_dense(g) for g in factors ]
    def gf_shoup(self, f):
        factors = gf_shoup(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
        return [ self.from_gf_dense(g) for g in factors ]

    def gf_factor_sqf(self, f, method=None):
        coeff, factors = gf_factor_sqf(self.to_gf_dense(f), self.domain.mod, self.domain.dom, method=method)
        return coeff, [ self.from_gf_dense(g) for g in factors ]
    def gf_factor(self, f):
        coeff, factors = gf_factor(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
        return coeff, [ (self.from_gf_dense(g), k) for g, k in factors ]
