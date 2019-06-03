import pytest
from pybind11_tests import pickling as m

try:
    import cPickle as pickle  # Use cPickle on Python 2.7
except ImportError:
    import pickle


@pytest.mark.parametrize("cls_name", ["Pickleable", "PickleableNew"])
def test_roundtrip(cls_name):
    cls = getattr(m, cls_name)
    p = cls("test_value")
    p.setExtra1(15)
    p.setExtra2(48)

    data = pickle.dumps(p, 2)  # Must use pickle protocol >= 2
    p2 = pickle.loads(data)
    assert p2.value() == p.value()
    assert p2.extra1() == p.extra1()
    assert p2.extra2() == p.extra2()


@pytest.unsupported_on_pypy
@pytest.mark.parametrize("cls_name", ["PickleableWithDict", "PickleableWithDictNew"])
def test_roundtrip_with_dict(cls_name):
    cls = getattr(m, cls_name)
    p = cls("test_value")
    p.extra = 15
    p.dynamic = "Attribute"

    data = pickle.dumps(p, pickle.HIGHEST_PROTOCOL)
    p2 = pickle.loads(data)
    assert p2.value == p.value
    assert p2.extra == p.extra
    assert p2.dynamic == p.dynamic


def test_enum_pickle():
    from pybind11_tests import enums as e
    data = pickle.dumps(e.EOne, 2)
    assert e.EOne == pickle.loads(data)
