from collections import defaultdict
from sympy.core.containers import Tuple
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.combinatorial.numbers import totient
from sympy.ntheory import n_order, is_primitive_root, is_quad_residue, \
    legendre_symbol, jacobi_symbol, primerange, sqrt_mod, \
    primitive_root, quadratic_residues, is_nthpow_residue, nthroot_mod, \
    sqrt_mod_iter, mobius, discrete_log, quadratic_congruence, \
    polynomial_congruence, sieve
from sympy.ntheory.residue_ntheory import _primitive_root_prime_iter, \
    _primitive_root_prime_power_iter, _primitive_root_prime_power2_iter, \
    _nthroot_mod_prime_power, _discrete_log_trial_mul, _discrete_log_shanks_steps, \
    _discrete_log_pollard_rho, _discrete_log_index_calculus, _discrete_log_pohlig_hellman, \
    _binomial_mod_prime_power, binomial_mod
from sympy.polys.domains import ZZ
from sympy.testing.pytest import raises
from sympy.core.random import randint, choice


def test_residue():
    assert n_order(2, 13) == 12
    assert [n_order(a, 7) for a in range(1, 7)] == \
           [1, 3, 6, 3, 6, 2]
    assert n_order(5, 17) == 16
    assert n_order(17, 11) == n_order(6, 11)
    assert n_order(101, 119) == 6
    assert n_order(11, (10**50 + 151)**2) == 10000000000000000000000000000000000000000000000030100000000000000000000000000000000000000000000022650
    raises(ValueError, lambda: n_order(6, 9))

    assert is_primitive_root(2, 7) is False
    assert is_primitive_root(3, 8) is False
    assert is_primitive_root(11, 14) is False
    assert is_primitive_root(12, 17) == is_primitive_root(29, 17)
    raises(ValueError, lambda: is_primitive_root(3, 6))

    for p in primerange(3, 100):
        li = list(_primitive_root_prime_iter(p))
        assert li[0] == min(li)
        for g in li:
            assert n_order(g, p) == p - 1
        assert len(li) == totient(totient(p))
        for e in range(1, 4):
            li_power = list(_primitive_root_prime_power_iter(p, e))
            li_power2 = list(_primitive_root_prime_power2_iter(p, e))
            assert len(li_power) == len(li_power2) == totient(totient(p**e))
    assert primitive_root(97) == 5
    assert n_order(primitive_root(97, False), 97) == totient(97)
    assert primitive_root(97**2) == 5
    assert n_order(primitive_root(97**2, False), 97**2) == totient(97**2)
    assert primitive_root(40487) == 5
    assert n_order(primitive_root(40487, False), 40487) == totient(40487)
    # note that primitive_root(40487) + 40487 = 40492 is a primitive root
    # of 40487**2, but it is not the smallest
    assert primitive_root(40487**2) == 10
    assert n_order(primitive_root(40487**2, False), 40487**2) == totient(40487**2)
    assert primitive_root(82) == 7
    assert n_order(primitive_root(82, False), 82) == totient(82)
    p = 10**50 + 151
    assert primitive_root(p) == 11
    assert n_order(primitive_root(p, False), p) == totient(p)
    assert primitive_root(2*p) == 11
    assert n_order(primitive_root(2*p, False), 2*p) == totient(2*p)
    assert primitive_root(p**2) == 11
    assert n_order(primitive_root(p**2, False), p**2) == totient(p**2)
    assert primitive_root(4 * 11) is None and primitive_root(4 * 11, False) is None
    assert primitive_root(15) is None and primitive_root(15, False) is None
    raises(ValueError, lambda: primitive_root(-3))

    assert is_quad_residue(3, 7) is False
    assert is_quad_residue(10, 13) is True
    assert is_quad_residue(12364, 139) == is_quad_residue(12364 % 139, 139)
    assert is_quad_residue(207, 251) is True
    assert is_quad_residue(0, 1) is True
    assert is_quad_residue(1, 1) is True
    assert is_quad_residue(0, 2) == is_quad_residue(1, 2) is True
    assert is_quad_residue(1, 4) is True
    assert is_quad_residue(2, 27) is False
    assert is_quad_residue(13122380800, 13604889600) is True
    assert [j for j in range(14) if is_quad_residue(j, 14)] == \
           [0, 1, 2, 4, 7, 8, 9, 11]
    raises(ValueError, lambda: is_quad_residue(1.1, 2))
    raises(ValueError, lambda: is_quad_residue(2, 0))

    assert quadratic_residues(S.One) == [0]
    assert quadratic_residues(1) == [0]
    assert quadratic_residues(12) == [0, 1, 4, 9]
    assert quadratic_residues(13) == [0, 1, 3, 4, 9, 10, 12]
    assert [len(quadratic_residues(i)) for i in range(1, 20)] == \
      [1, 2, 2, 2, 3, 4, 4, 3, 4, 6, 6, 4, 7, 8, 6, 4, 9, 8, 10]

    assert list(sqrt_mod_iter(6, 2)) == [0]
    assert sqrt_mod(3, 13) == 4
    assert sqrt_mod(3, -13) == 4
    assert sqrt_mod(6, 23) == 11
    assert sqrt_mod(345, 690) == 345
    assert sqrt_mod(67, 101) == None
    assert sqrt_mod(1020, 104729) == None

    for p in range(3, 100):
        d = defaultdict(list)
        for i in range(p):
            d[pow(i, 2, p)].append(i)
        for i in range(1, p):
            it = sqrt_mod_iter(i, p)
            v = sqrt_mod(i, p, True)
            if v:
                v = sorted(v)
                assert d[i] == v
            else:
                assert not d[i]

    assert sqrt_mod(9, 27, True) == [3, 6, 12, 15, 21, 24]
    assert sqrt_mod(9, 81, True) == [3, 24, 30, 51, 57, 78]
    assert sqrt_mod(9, 3**5, True) == [3, 78, 84, 159, 165, 240]
    assert sqrt_mod(81, 3**4, True) == [0, 9, 18, 27, 36, 45, 54, 63, 72]
    assert sqrt_mod(81, 3**5, True) == [9, 18, 36, 45, 63, 72, 90, 99, 117,\
            126, 144, 153, 171, 180, 198, 207, 225, 234]
    assert sqrt_mod(81, 3**6, True) == [9, 72, 90, 153, 171, 234, 252, 315,\
            333, 396, 414, 477, 495, 558, 576, 639, 657, 720]
    assert sqrt_mod(81, 3**7, True) == [9, 234, 252, 477, 495, 720, 738, 963,\
            981, 1206, 1224, 1449, 1467, 1692, 1710, 1935, 1953, 2178]

    for a, p in [(26214400, 32768000000), (26214400, 16384000000),
        (262144, 1048576), (87169610025, 163443018796875),
        (22315420166400, 167365651248000000)]:
        assert pow(sqrt_mod(a, p), 2, p) == a

    n = 70
    a, p = 5**2*3**n*2**n, 5**6*3**(n+1)*2**(n+2)
    it = sqrt_mod_iter(a, p)
    for i in range(10):
        assert pow(next(it), 2, p) == a
    a, p = 5**2*3**n*2**n, 5**6*3**(n+1)*2**(n+3)
    it = sqrt_mod_iter(a, p)
    for i in range(2):
        assert pow(next(it), 2, p) == a
    n = 100
    a, p = 5**2*3**n*2**n, 5**6*3**(n+1)*2**(n+1)
    it = sqrt_mod_iter(a, p)
    for i in range(2):
        assert pow(next(it), 2, p) == a

    assert type(next(sqrt_mod_iter(9, 27))) is int
    assert type(next(sqrt_mod_iter(9, 27, ZZ))) is type(ZZ(1))
    assert type(next(sqrt_mod_iter(1, 7, ZZ))) is type(ZZ(1))

    assert is_nthpow_residue(2, 1, 5)

    #issue 10816
    assert is_nthpow_residue(1, 0, 1) is False
    assert is_nthpow_residue(1, 0, 2) is True
    assert is_nthpow_residue(3, 0, 2) is True
    assert is_nthpow_residue(0, 1, 8) is True
    assert is_nthpow_residue(2, 3, 2) is True
    assert is_nthpow_residue(2, 3, 9) is False
    assert is_nthpow_residue(3, 5, 30) is True
    assert is_nthpow_residue(21, 11, 20) is True
    assert is_nthpow_residue(7, 10, 20) is False
    assert is_nthpow_residue(5, 10, 20) is True
    assert is_nthpow_residue(3, 10, 48) is False
    assert is_nthpow_residue(1, 10, 40) is True
    assert is_nthpow_residue(3, 10, 24) is False
    assert is_nthpow_residue(1, 10, 24) is True
    assert is_nthpow_residue(3, 10, 24) is False
    assert is_nthpow_residue(2, 10, 48) is False
    assert is_nthpow_residue(81, 3, 972) is False
    assert is_nthpow_residue(243, 5, 5103) is True
    assert is_nthpow_residue(243, 3, 1240029) is False
    assert is_nthpow_residue(36010, 8, 87382) is True
    assert is_nthpow_residue(28552, 6, 2218) is True
    assert is_nthpow_residue(92712, 9, 50026) is True
    x = {pow(i, 56, 1024) for i in range(1024)}
    assert {a for a in range(1024) if is_nthpow_residue(a, 56, 1024)} == x
    x = { pow(i, 256, 2048) for i in range(2048)}
    assert {a for a in range(2048) if is_nthpow_residue(a, 256, 2048)} == x
    x = { pow(i, 11, 324000) for i in range(1000)}
    assert [ is_nthpow_residue(a, 11, 324000) for a in x]
    x = { pow(i, 17, 22217575536) for i in range(1000)}
    assert [ is_nthpow_residue(a, 17, 22217575536) for a in x]
    assert is_nthpow_residue(676, 3, 5364)
    assert is_nthpow_residue(9, 12, 36)
    assert is_nthpow_residue(32, 10, 41)
    assert is_nthpow_residue(4, 2, 64)
    assert is_nthpow_residue(31, 4, 41)
    assert not is_nthpow_residue(2, 2, 5)
    assert is_nthpow_residue(8547, 12, 10007)
    assert is_nthpow_residue(Dummy(even=True) + 3, 3, 2) == True
    # _nthroot_mod_prime_power
    for p in primerange(2, 10):
        for a in range(3):
            for n in range(3, 5):
                ans = _nthroot_mod_prime_power(a, n, p, 1)
                assert isinstance(ans, list)
                if len(ans) == 0:
                    for b in range(p):
                        assert pow(b, n, p) != a % p
                    for k in range(2, 10):
                        assert _nthroot_mod_prime_power(a, n, p, k) == []
                else:
                    for b in range(p):
                        pred = pow(b, n, p) == a % p
                        assert not(pred ^ (b in ans))
                    for k in range(2, 10):
                        ans = _nthroot_mod_prime_power(a, n, p, k)
                        if not ans:
                            break
                        for b in ans:
                            assert pow(b, n , p**k) == a

    assert nthroot_mod(Dummy(odd=True), 3, 2) == 1
    assert nthroot_mod(29, 31, 74) == 45
    assert nthroot_mod(1801, 11, 2663) == 44
    for a, q, p in [(51922, 2, 203017), (43, 3, 109), (1801, 11, 2663),
          (26118163, 1303, 33333347), (1499, 7, 2663), (595, 6, 2663),
          (1714, 12, 2663), (28477, 9, 33343)]:
        r = nthroot_mod(a, q, p)
        assert pow(r, q, p) == a
    assert nthroot_mod(11, 3, 109) is None
    assert nthroot_mod(16, 5, 36, True) == [4, 22]
    assert nthroot_mod(9, 16, 36, True) == [3, 9, 15, 21, 27, 33]
    assert nthroot_mod(4, 3, 3249000) is None
    assert nthroot_mod(36010, 8, 87382, True) == [40208, 47174]
    assert nthroot_mod(0, 12, 37, True) == [0]
    assert nthroot_mod(0, 7, 100, True) == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    assert nthroot_mod(4, 4, 27, True) == [5, 22]
    assert nthroot_mod(4, 4, 121, True) == [19, 102]
    assert nthroot_mod(2, 3, 7, True) == []
    for p in range(1, 20):
        for a in range(p):
            for n in range(1, p):
                ans = nthroot_mod(a, n, p, True)
                assert isinstance(ans, list)
                for b in range(p):
                    pred = pow(b, n, p) == a
                    assert not(pred ^ (b in ans))
                ans2 = nthroot_mod(a, n, p, False)
                if ans2 is None:
                    assert ans == []
                else:
                    assert ans2 in ans

    x = Symbol('x', positive=True)
    i = Symbol('i', integer=True)
    assert _discrete_log_trial_mul(587, 2**7, 2) == 7
    assert _discrete_log_trial_mul(941, 7**18, 7) == 18
    assert _discrete_log_trial_mul(389, 3**81, 3) == 81
    assert _discrete_log_trial_mul(191, 19**123, 19) == 123
    assert _discrete_log_shanks_steps(442879, 7**2, 7) == 2
    assert _discrete_log_shanks_steps(874323, 5**19, 5) == 19
    assert _discrete_log_shanks_steps(6876342, 7**71, 7) == 71
    assert _discrete_log_shanks_steps(2456747, 3**321, 3) == 321
    assert _discrete_log_pollard_rho(6013199, 2**6, 2, rseed=0) == 6
    assert _discrete_log_pollard_rho(6138719, 2**19, 2, rseed=0) == 19
    assert _discrete_log_pollard_rho(36721943, 2**40, 2, rseed=0) == 40
    assert _discrete_log_pollard_rho(24567899, 3**333, 3, rseed=0) == 333
    raises(ValueError, lambda: _discrete_log_pollard_rho(11, 7, 31, rseed=0))
    raises(ValueError, lambda: _discrete_log_pollard_rho(227, 3**7, 5, rseed=0))
    assert _discrete_log_index_calculus(983, 948, 2, 491) == 183
    assert _discrete_log_index_calculus(633383, 21794, 2, 316691) == 68048
    assert _discrete_log_index_calculus(941762639, 68822582, 2, 470881319) == 338029275
    assert _discrete_log_index_calculus(999231337607, 888188918786, 2, 499615668803) == 142811376514
    assert _discrete_log_index_calculus(47747730623, 19410045286, 43425105668, 645239603) == 590504662
    assert _discrete_log_pohlig_hellman(98376431, 11**9, 11) == 9
    assert _discrete_log_pohlig_hellman(78723213, 11**31, 11) == 31
    assert _discrete_log_pohlig_hellman(32942478, 11**98, 11) == 98
    assert _discrete_log_pohlig_hellman(14789363, 11**444, 11) == 444
    assert discrete_log(1, 0, 2) == 0
    raises(ValueError, lambda: discrete_log(-4, 1, 3))
    raises(ValueError, lambda: discrete_log(10, 3, 2))
    assert discrete_log(587, 2**9, 2) == 9
    assert discrete_log(2456747, 3**51, 3) == 51
    assert discrete_log(32942478, 11**127, 11) == 127
    assert discrete_log(432751500361, 7**324, 7) == 324
    assert discrete_log(265390227570863,184500076053622, 2) == 17835221372061
    assert discrete_log(22708823198678103974314518195029102158525052496759285596453269189798311427475159776411276642277139650833937,
                        17463946429475485293747680247507700244427944625055089103624311227422110546803452417458985046168310373075327,
                        123456) == 2068031853682195777930683306640554533145512201725884603914601918777510185469769997054750835368413389728895
    args = 5779, 3528, 6215
    assert discrete_log(*args) == 687
    assert discrete_log(*Tuple(*args)) == 687
    assert quadratic_congruence(400, 85, 125, 1600) == [295, 615, 935, 1255, 1575]
    assert quadratic_congruence(3, 6, 5, 25) == [3, 20]
    assert quadratic_congruence(120, 80, 175, 500) == []
    assert quadratic_congruence(15, 14, 7, 2) == [1]
    assert quadratic_congruence(8, 15, 7, 29) == [10, 28]
    assert quadratic_congruence(160, 200, 300, 461) == [144, 431]
    assert quadratic_congruence(100000, 123456, 7415263, 48112959837082048697) == [30417843635344493501, 36001135160550533083]
    assert quadratic_congruence(65, 121, 72, 277) == [249, 252]
    assert quadratic_congruence(5, 10, 14, 2) == [0]
    assert quadratic_congruence(10, 17, 19, 2) == [1]
    assert quadratic_congruence(10, 14, 20, 2) == [0, 1]
    assert quadratic_congruence(2**48-7, 2**48-1, 4, 2**48) == [8249717183797, 31960993774868]
    assert polynomial_congruence(6*x**5 + 10*x**4 + 5*x**3 + x**2 + x + 1,
        972000) == [220999, 242999, 463999, 485999, 706999, 728999, 949999, 971999]

    assert polynomial_congruence(x**3 - 10*x**2 + 12*x - 82, 33075) == [30287]
    assert polynomial_congruence(x**2 + x + 47, 2401) == [785, 1615]
    assert polynomial_congruence(10*x**2 + 14*x + 20, 2) == [0, 1]
    assert polynomial_congruence(x**3 + 3, 16) == [5]
    assert polynomial_congruence(65*x**2 + 121*x + 72, 277) == [249, 252]
    assert polynomial_congruence(x**4 - 4, 27) == [5, 22]
    assert polynomial_congruence(35*x**3 - 6*x**2 - 567*x + 2308, 148225) == [86957,
        111157, 122531, 146731]
    assert polynomial_congruence(x**16 - 9, 36) == [3, 9, 15, 21, 27, 33]
    assert polynomial_congruence(x**6 - 2*x**5 - 35, 6125) == [3257]
    raises(ValueError, lambda: polynomial_congruence(x**x, 6125))
    raises(ValueError, lambda: polynomial_congruence(x**i, 6125))
    raises(ValueError, lambda: polynomial_congruence(0.1*x**2 + 6, 100))

    assert binomial_mod(-1, 1, 10) == 0
    assert binomial_mod(1, -1, 10) == 0
    raises(ValueError, lambda: binomial_mod(2, 1, -1))
    assert binomial_mod(51, 10, 10) == 0
    assert binomial_mod(10**3, 500, 3**6) == 567
    assert binomial_mod(10**18 - 1, 123456789, 4) == 0
    assert binomial_mod(10**18, 10**12, (10**5 + 3)**2) == 3744312326


def test_binomial_p_pow():
    n, binomials, binomial = 1000, [1], 1
    for i in range(1, n + 1):
        binomial *= n - i + 1
        binomial //= i
        binomials.append(binomial)

    # Test powers of two, which the algorithm treats slightly differently
    trials_2 = 100
    for _ in range(trials_2):
        m, power = randint(0, n), randint(1, 20)
        assert _binomial_mod_prime_power(n, m, 2, power) == binomials[m] % 2**power

    # Test against other prime powers
    primes = list(sieve.primerange(2*n))
    trials = 1000
    for _ in range(trials):
        m, prime, power = randint(0, n), choice(primes), randint(1, 10)
        assert _binomial_mod_prime_power(n, m, prime, power) == binomials[m] % prime**power


def test_deprecated_ntheory_symbolic_functions():
    from sympy.testing.pytest import warns_deprecated_sympy

    with warns_deprecated_sympy():
        assert mobius(3) == -1
    with warns_deprecated_sympy():
        assert legendre_symbol(2, 3) == -1
    with warns_deprecated_sympy():
        assert jacobi_symbol(2, 3) == -1
