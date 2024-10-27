from sympy.multipledispatch.conflict import (supercedes, ordering, ambiguities,
        ambiguous, super_signature, consistent)


class A: pass
class B(A): pass
class C: pass


def test_supercedes():
    assert supercedes([B], [A])
    assert supercedes([B, A], [A, A])
    assert not supercedes([B, A], [A, B])
    assert not supercedes([A], [B])


def test_consistent():
    assert consistent([A], [A])
    assert consistent([B], [B])
    assert not consistent([A], [C])
    assert consistent([A, B], [A, B])
    assert consistent([B, A], [A, B])
    assert not consistent([B, A], [B])
    assert not consistent([B, A], [B, C])


def test_super_signature():
    assert super_signature([[A]]) == [A]
    assert super_signature([[A], [B]]) == [B]
    assert super_signature([[A, B], [B, A]]) == [B, B]
    assert super_signature([[A, A, B], [A, B, A], [B, A, A]]) == [B, B, B]


def test_ambiguous():
    assert not ambiguous([A], [A])
    assert not ambiguous([A], [B])
    assert not ambiguous([B], [B])
    assert not ambiguous([A, B], [B, B])
    assert ambiguous([A, B], [B, A])


def test_ambiguities():
    signatures = [[A], [B], [A, B], [B, A], [A, C]]
    expected = {((A, B), (B, A))}
    result = ambiguities(signatures)
    assert set(map(frozenset, expected)) == set(map(frozenset, result))

    signatures = [[A], [B], [A, B], [B, A], [A, C], [B, B]]
    expected = set()
    result = ambiguities(signatures)
    assert set(map(frozenset, expected)) == set(map(frozenset, result))


def test_ordering():
    signatures = [[A, A], [A, B], [B, A], [B, B], [A, C]]
    ord = ordering(signatures)
    assert ord[0] == (B, B) or ord[0] == (A, C)
    assert ord[-1] == (A, A) or ord[-1] == (A, C)


def test_type_mro():
    assert super_signature([[object], [type]]) == [type]
