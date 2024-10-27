from sympy.ntheory.partitions_ import npartitions, _partition_rec, _partition


def test__partition_rec():
    A000041 = [1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56, 77, 101, 135,
               176, 231, 297, 385, 490, 627, 792, 1002, 1255, 1575]
    for n, val in enumerate(A000041):
        assert _partition_rec(n) == val


def test__partition():
    assert [_partition(k) for k in range(13)] == \
        [1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56, 77]
    assert _partition(100) == 190569292
    assert _partition(200) == 3972999029388
    assert _partition(1000) == 24061467864032622473692149727991
    assert _partition(1001) == 25032297938763929621013218349796
    assert _partition(2000) == 4720819175619413888601432406799959512200344166
    assert _partition(10000) % 10**10 == 6916435144
    assert _partition(100000) % 10**10 == 9421098519


def test_deprecated_ntheory_symbolic_functions():
    from sympy.testing.pytest import warns_deprecated_sympy

    with warns_deprecated_sympy():
        assert npartitions(0) == 1
