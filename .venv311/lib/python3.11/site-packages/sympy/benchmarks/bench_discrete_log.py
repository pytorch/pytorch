import sys
from time import time
from sympy.ntheory.residue_ntheory import (discrete_log,
        _discrete_log_trial_mul, _discrete_log_shanks_steps,
        _discrete_log_pollard_rho, _discrete_log_pohlig_hellman)


# Cyclic group (Z/pZ)* with p prime, order p - 1 and generator g
data_set_1 = [
        # p, p - 1, g
        [191, 190, 19],
        [46639, 46638, 6],
        [14789363, 14789362, 2],
        [4254225211, 4254225210, 2],
        [432751500361, 432751500360, 7],
        [158505390797053, 158505390797052, 2],
        [6575202655312007, 6575202655312006, 5],
        [8430573471995353769, 8430573471995353768, 3],
        [3938471339744997827267, 3938471339744997827266, 2],
        [875260951364705563393093, 875260951364705563393092, 5],
    ]


# Cyclic sub-groups of (Z/nZ)* with prime order p and generator g
# (n, p are primes and n = 2 * p + 1)
data_set_2 = [
        # n, p, g
        [227, 113, 3],
        [2447, 1223, 2],
        [24527, 12263, 2],
        [245639, 122819, 2],
        [2456747, 1228373, 3],
        [24567899, 12283949, 3],
        [245679023, 122839511, 2],
        [2456791307, 1228395653, 3],
        [24567913439, 12283956719, 2],
        [245679135407, 122839567703, 2],
        [2456791354763, 1228395677381, 3],
        [24567913550903, 12283956775451, 2],
        [245679135509519, 122839567754759, 2],
    ]


# Cyclic sub-groups of (Z/nZ)* with smooth order o and generator g
data_set_3 = [
        # n, o, g
        [2**118, 2**116, 3],
    ]


def bench_discrete_log(data_set, algo=None):
    if algo is None:
        f = discrete_log
    elif algo == 'trial':
        f = _discrete_log_trial_mul
    elif algo == 'shanks':
        f = _discrete_log_shanks_steps
    elif algo == 'rho':
        f = _discrete_log_pollard_rho
    elif algo == 'ph':
        f = _discrete_log_pohlig_hellman
    else:
        raise ValueError("Argument 'algo' should be one"
                " of ('trial', 'shanks', 'rho' or 'ph')")

    for i, data in enumerate(data_set):
        for j, (n, p, g) in enumerate(data):
            t = time()
            l = f(n, pow(g, p - 1, n), g, p)
            t = time() - t
            print('[%02d-%03d] %15.10f' % (i, j, t))
            assert l == p - 1


if __name__ == '__main__':
    algo = sys.argv[1] \
            if len(sys.argv) > 1 else None
    data_set = [
            data_set_1,
            data_set_2,
            data_set_3,
        ]
    bench_discrete_log(data_set, algo)
