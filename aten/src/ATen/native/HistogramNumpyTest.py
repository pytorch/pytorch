import numpy as np
import torch as t
import random
import time
import statistics
import math

# TODO: figure out why we get inexact matches when passing weights
TOL = 1e-12

TEST_HISTC = False

INPUT_SZ = int(1e6)
BIN_CT = int(1e2)

iterations = 0
times = {}

def record_times(dtype, density, weight, bin_edges, np_t, t_t):
    key = (dtype, \
            'density_yes' if density else 'density_no', \
            'weighted' if weight else 'unweighted', \
            'bin_edges' if not isinstance(bin_edges, int) else 'bin_ct')

    if key not in times:
        times[key] = ([], [])

    times[key][0].append(np_t)
    times[key][1].append(t_t)

def print_list_summary(a):
    stdev = 0 if len(a) < 2 else statistics.stdev(a)
    print('%.15f %.15f %.15f %.15f ' % (sum(a)/len(a), min(a), max(a), stdev), end = '')

def print_times():
    global iterations
    iterations = iterations + 1

    print("--- SUMMARY AFTER", iterations, " ITERATIONS ---")
    for key in times:
        (np_times, t_times) = times[key]

        (dtype, density, weight, bins) = key
        print(str(dtype)[6:], density, weight, bins, end=' ')
        print_list_summary(np_times)
        print_list_summary(t_times)
        print()

def check(x, xt, y, yt, density, weight):
    # gate for histc
    if TEST_HISTC and (xt.dtype == t.int32 or density or weight or not isinstance(yt, int)):
        return

    npres = None
    tres = None
    res_within_tol = True
    bins_within_tol = True

    try:
        weight_t = None if not weight else t.tensor(weight, dtype=t.float64)

        xn = np.array(x)
        yn = np.array(y)

        if iterations % 2 == 0:
            npt0 = time.time()
            np_out = np.histogram(xn, yn, density=density, weights=weight)
            npt1 = time.time()

            tt0 = time.time()
            if TEST_HISTC:
                torch_out = t.histc(xt, yt)
            else:
                torch_out = t.histogram(xt, yt, density=density, weight=weight_t)
            tt1 = time.time()
        else:
            tt0 = time.time()
            if TEST_HISTC:
                torch_out = t.histc(xt, yt)
            else:
                torch_out = t.histogram(xt, yt, density=density, weight=weight_t)

            tt1 = time.time()

            npt0 = time.time()
            np_out = np.histogram(xn, yn, density=density, weights=weight)
            npt1 = time.time()

        record_times(xt.dtype, density, weight, y, npt1 - npt0, tt1 - tt0)

        npres = np_out[0].tolist()
        npbins = np_out[1].tolist()

        tres = torch_out.tolist() if TEST_HISTC else torch_out.hist.tolist()
        tbins = npbins if TEST_HISTC else torch_out.bin_edges.tolist()

        """
        # make sure we get the same output if we pass empty tensors to the out version
        hist_out = t.tensor([], dtype=torch_out.hist.dtype);
        bins_out = t.tensor([], dtype=torch_out.bin_edges.dtype);
        t.histogram(xt, yt, density=density, weight=weight_t, out=(hist_out, bins_out))
        if (hist_out.tolist(), bins_out.tolist()) != (tres, tbins):
            raise Exception("Got hist =", tres, " bins =", tbins, " but then the 'out' version ", \
                    "produced hist =", hist_out.tolist(), " bins =", bins_out.tolist())

        # make sure we get the same output if we pass correctly-sized tensors to the out version
        # (just reuse hist_out and bins_out)
        t.histogram(xt, yt, density=density, weight=weight_t, out=(hist_out, bins_out))
        if (hist_out.tolist(), bins_out.tolist()) != (tres, tbins):
            raise Exception("Got hist =", tres, " bins =", tbins, " but then the 'out' version ", \
                    "produced hist =", hist_out.tolist(), " bins =", bins_out.tolist())
        """

        def difference(x, y):
            if math.isnan(x) and math.isnan(y):
                return 0
            return abs(x - y)

        res_max_diff = max([difference(npres[i], tres[i]) for i in range(len(tres))])
        bins_max_diff = max([difference(npbins[i], tbins[i]) for i in range(len(tbins))])

        res_within_tol = len(npres) == len(tres) and res_max_diff < TOL
        bins_within_tol = len(npbins) == len(tbins) and bins_max_diff < TOL

        print('(dtype =', xt.dtype, ', density =', density, \
                ', weighted =', 'True' if weight else 'False', \
                ', explicit_bin_edges =', 'False' if isinstance(y, int) else 'True', \
                ', hist_exact =', npres == tres,  \
                ', bins_exact =', npbins == tbins, ') ', end = '')

        if res_within_tol and bins_within_tol:
            print('OK')
            #print(npres, npbins)
            #print(tres, tbins)
        else:
            print('bad!!!')

            print("input =", x)
            print("bins =", y)
            print("density =", density)
            print("weight =", weight)

            print("np hist =", npres)
            print("np bins =", npbins)

            print("t hist =", tres)
            print("t bins =", tbins)

            print("worst hist diff is", res_max_diff)
            print("worst bin_edge diff is", bins_max_diff)

            exit(0)

    except Exception as e:
        print(e)
        print('encountered error (dtype =', xt.dtype, ', density =', density, ', weighted =', 'True' if weight else 'False', ')')
        exit(0)

def rand_float_test(density, weight):
    x = [random.uniform(-1, 1) for i in range(0, INPUT_SZ)]
    xt = t.tensor(x, dtype=t.float64) # numpy arrays default to float64

    # just pass number of bins
    #bin_ct = random.randint(1, BIN_CT + 1)
    bin_ct = BIN_CT
    check(x, xt, bin_ct, bin_ct, density, None if not weight else [random.uniform(0, 1) for elt in x])

    # or construct bin edges ourselves
    y = [random.uniform(-1, 1) for i in range(0, BIN_CT)]
    y.sort()
    yt = t.tensor(y, dtype=t.float64)
    check(x, xt, y, yt, density, None if not weight else [random.uniform(0, 1) for elt in x])

def rand_int_test(density, weight):
    x = [random.randint(-100, 100) for i in range(0, INPUT_SZ)]
    xt = t.tensor(x, dtype=t.int32)

    # just pass number of bins
    #bin_ct = random.randint(1, BIN_CT + 1)
    bin_ct = BIN_CT
    check(x, xt, bin_ct, bin_ct, density, None if not weight else [random.uniform(0, 1) for elt in x])

    # construct bin edges ourselves
    y = list(set([random.randint(-100, 100) for i in range(0, BIN_CT)]))
    y.sort()
    yt = t.tensor(y, dtype=t.int32)
    check(x, xt, y, yt, density, None if not weight else [random.uniform(0, 1) for elt in x])

while iterations < 25:
    if TEST_HISTC:
        rand_float_test(False, False)
    else:
        for density in [False, True]:
            for weight in [False, True]:
                rand_float_test(density = density, weight = weight)
                rand_int_test(density = density, weight = weight)

    print_times()

