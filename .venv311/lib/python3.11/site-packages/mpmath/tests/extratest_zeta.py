from mpmath import zetazero
from timeit import default_timer as clock

def test_zetazero():
    cases = [\
    (399999999, 156762524.6750591511),
    (241389216, 97490234.2276711795),
    (526196239, 202950727.691229534),
    (542964976, 209039046.578535272),
    (1048449112, 388858885.231056486),
    (1048449113, 388858885.384337406),
    (1048449114, 388858886.002285122),
    (1048449115, 388858886.00239369),
    (1048449116, 388858886.690745053)
    ]
    for n, v in cases:
        print(n, v)
        t1 = clock()
        ok = zetazero(n).ae(complex(0.5,v))
        t2 = clock()
        print("ok =", ok, ("(time = %s)" % round(t2-t1,3)))
    print("Now computing two huge zeros (this may take hours)")
    print("Computing zetazero(8637740722917)")
    ok = zetazero(8637740722917).ae(complex(0.5,2124447368584.39296466152))
    print("ok =", ok)
    ok = zetazero(8637740722918).ae(complex(0.5,2124447368584.39298170604))
    print("ok =", ok)

if __name__ == "__main__":
    test_zetazero()
