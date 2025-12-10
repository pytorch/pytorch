from sympy.categories import (Object, Morphism, IdentityMorphism,
                              NamedMorphism, CompositeMorphism,
                              Diagram, Category)
from sympy.categories.baseclasses import Class
from sympy.testing.pytest import raises
from sympy.core.containers import (Dict, Tuple)
from sympy.sets import EmptySet
from sympy.sets.sets import FiniteSet


def test_morphisms():
    A = Object("A")
    B = Object("B")
    C = Object("C")
    D = Object("D")

    # Test the base morphism.
    f = NamedMorphism(A, B, "f")
    assert f.domain == A
    assert f.codomain == B
    assert f == NamedMorphism(A, B, "f")

    # Test identities.
    id_A = IdentityMorphism(A)
    id_B = IdentityMorphism(B)
    assert id_A.domain == A
    assert id_A.codomain == A
    assert id_A == IdentityMorphism(A)
    assert id_A != id_B

    # Test named morphisms.
    g = NamedMorphism(B, C, "g")
    assert g.name == "g"
    assert g != f
    assert g == NamedMorphism(B, C, "g")
    assert g != NamedMorphism(B, C, "f")

    # Test composite morphisms.
    assert f == CompositeMorphism(f)

    k = g.compose(f)
    assert k.domain == A
    assert k.codomain == C
    assert k.components == Tuple(f, g)
    assert g * f == k
    assert CompositeMorphism(f, g) == k

    assert CompositeMorphism(g * f) == g * f

    # Test the associativity of composition.
    h = NamedMorphism(C, D, "h")

    p = h * g
    u = h * g * f

    assert h * k == u
    assert p * f == u
    assert CompositeMorphism(f, g, h) == u

    # Test flattening.
    u2 = u.flatten("u")
    assert isinstance(u2, NamedMorphism)
    assert u2.name == "u"
    assert u2.domain == A
    assert u2.codomain == D

    # Test identities.
    assert f * id_A == f
    assert id_B * f == f
    assert id_A * id_A == id_A
    assert CompositeMorphism(id_A) == id_A

    # Test bad compositions.
    raises(ValueError, lambda: f * g)

    raises(TypeError, lambda: f.compose(None))
    raises(TypeError, lambda: id_A.compose(None))
    raises(TypeError, lambda: f * None)
    raises(TypeError, lambda: id_A * None)

    raises(TypeError, lambda: CompositeMorphism(f, None, 1))

    raises(ValueError, lambda: NamedMorphism(A, B, ""))
    raises(NotImplementedError, lambda: Morphism(A, B))


def test_diagram():
    A = Object("A")
    B = Object("B")
    C = Object("C")

    f = NamedMorphism(A, B, "f")
    g = NamedMorphism(B, C, "g")
    id_A = IdentityMorphism(A)
    id_B = IdentityMorphism(B)

    empty = EmptySet

    # Test the addition of identities.
    d1 = Diagram([f])

    assert d1.objects == FiniteSet(A, B)
    assert d1.hom(A, B) == (FiniteSet(f), empty)
    assert d1.hom(A, A) == (FiniteSet(id_A), empty)
    assert d1.hom(B, B) == (FiniteSet(id_B), empty)

    assert d1 == Diagram([id_A, f])
    assert d1 == Diagram([f, f])

    # Test the addition of composites.
    d2 = Diagram([f, g])
    homAC = d2.hom(A, C)[0]

    assert d2.objects == FiniteSet(A, B, C)
    assert g * f in d2.premises.keys()
    assert homAC == FiniteSet(g * f)

    # Test equality, inequality and hash.
    d11 = Diagram([f])

    assert d1 == d11
    assert d1 != d2
    assert hash(d1) == hash(d11)

    d11 = Diagram({f: "unique"})
    assert d1 != d11

    # Make sure that (re-)adding composites (with new properties)
    # works as expected.
    d = Diagram([f, g], {g * f: "unique"})
    assert d.conclusions == Dict({g * f: FiniteSet("unique")})

    # Check the hom-sets when there are premises and conclusions.
    assert d.hom(A, C) == (FiniteSet(g * f), FiniteSet(g * f))
    d = Diagram([f, g], [g * f])
    assert d.hom(A, C) == (FiniteSet(g * f), FiniteSet(g * f))

    # Check how the properties of composite morphisms are computed.
    d = Diagram({f: ["unique", "isomorphism"], g: "unique"})
    assert d.premises[g * f] == FiniteSet("unique")

    # Check that conclusion morphisms with new objects are not allowed.
    d = Diagram([f], [g])
    assert d.conclusions == Dict({})

    # Test an empty diagram.
    d = Diagram()
    assert d.premises == Dict({})
    assert d.conclusions == Dict({})
    assert d.objects == empty

    # Check a SymPy Dict object.
    d = Diagram(Dict({f: FiniteSet("unique", "isomorphism"), g: "unique"}))
    assert d.premises[g * f] == FiniteSet("unique")

    # Check the addition of components of composite morphisms.
    d = Diagram([g * f])
    assert f in d.premises
    assert g in d.premises

    # Check subdiagrams.
    d = Diagram([f, g], {g * f: "unique"})

    d1 = Diagram([f])
    assert d.is_subdiagram(d1)
    assert not d1.is_subdiagram(d)

    d = Diagram([NamedMorphism(B, A, "f'")])
    assert not d.is_subdiagram(d1)
    assert not d1.is_subdiagram(d)

    d1 = Diagram([f, g], {g * f: ["unique", "something"]})
    assert not d.is_subdiagram(d1)
    assert not d1.is_subdiagram(d)

    d = Diagram({f: "blooh"})
    d1 = Diagram({f: "bleeh"})
    assert not d.is_subdiagram(d1)
    assert not d1.is_subdiagram(d)

    d = Diagram([f, g], {f: "unique", g * f: "veryunique"})
    d1 = d.subdiagram_from_objects(FiniteSet(A, B))
    assert d1 == Diagram([f], {f: "unique"})
    raises(ValueError, lambda: d.subdiagram_from_objects(FiniteSet(A,
           Object("D"))))

    raises(ValueError, lambda: Diagram({IdentityMorphism(A): "unique"}))


def test_category():
    A = Object("A")
    B = Object("B")
    C = Object("C")

    f = NamedMorphism(A, B, "f")
    g = NamedMorphism(B, C, "g")

    d1 = Diagram([f, g])
    d2 = Diagram([f])

    objects = d1.objects | d2.objects

    K = Category("K", objects, commutative_diagrams=[d1, d2])

    assert K.name == "K"
    assert K.objects == Class(objects)
    assert K.commutative_diagrams == FiniteSet(d1, d2)

    raises(ValueError, lambda: Category(""))
