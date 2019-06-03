import pytest
from pybind11_tests import smart_ptr as m
from pybind11_tests import ConstructorStats


def test_smart_ptr(capture):
    # Object1
    for i, o in enumerate([m.make_object_1(), m.make_object_2(), m.MyObject1(3)], start=1):
        assert o.getRefCount() == 1
        with capture:
            m.print_object_1(o)
            m.print_object_2(o)
            m.print_object_3(o)
            m.print_object_4(o)
        assert capture == "MyObject1[{i}]\n".format(i=i) * 4

    for i, o in enumerate([m.make_myobject1_1(), m.make_myobject1_2(), m.MyObject1(6), 7],
                          start=4):
        print(o)
        with capture:
            if not isinstance(o, int):
                m.print_object_1(o)
                m.print_object_2(o)
                m.print_object_3(o)
                m.print_object_4(o)
            m.print_myobject1_1(o)
            m.print_myobject1_2(o)
            m.print_myobject1_3(o)
            m.print_myobject1_4(o)
        assert capture == "MyObject1[{i}]\n".format(i=i) * (4 if isinstance(o, int) else 8)

    cstats = ConstructorStats.get(m.MyObject1)
    assert cstats.alive() == 0
    expected_values = ['MyObject1[{}]'.format(i) for i in range(1, 7)] + ['MyObject1[7]'] * 4
    assert cstats.values() == expected_values
    assert cstats.default_constructions == 0
    assert cstats.copy_constructions == 0
    # assert cstats.move_constructions >= 0 # Doesn't invoke any
    assert cstats.copy_assignments == 0
    assert cstats.move_assignments == 0

    # Object2
    for i, o in zip([8, 6, 7], [m.MyObject2(8), m.make_myobject2_1(), m.make_myobject2_2()]):
        print(o)
        with capture:
            m.print_myobject2_1(o)
            m.print_myobject2_2(o)
            m.print_myobject2_3(o)
            m.print_myobject2_4(o)
        assert capture == "MyObject2[{i}]\n".format(i=i) * 4

    cstats = ConstructorStats.get(m.MyObject2)
    assert cstats.alive() == 1
    o = None
    assert cstats.alive() == 0
    assert cstats.values() == ['MyObject2[8]', 'MyObject2[6]', 'MyObject2[7]']
    assert cstats.default_constructions == 0
    assert cstats.copy_constructions == 0
    # assert cstats.move_constructions >= 0 # Doesn't invoke any
    assert cstats.copy_assignments == 0
    assert cstats.move_assignments == 0

    # Object3
    for i, o in zip([9, 8, 9], [m.MyObject3(9), m.make_myobject3_1(), m.make_myobject3_2()]):
        print(o)
        with capture:
            m.print_myobject3_1(o)
            m.print_myobject3_2(o)
            m.print_myobject3_3(o)
            m.print_myobject3_4(o)
        assert capture == "MyObject3[{i}]\n".format(i=i) * 4

    cstats = ConstructorStats.get(m.MyObject3)
    assert cstats.alive() == 1
    o = None
    assert cstats.alive() == 0
    assert cstats.values() == ['MyObject3[9]', 'MyObject3[8]', 'MyObject3[9]']
    assert cstats.default_constructions == 0
    assert cstats.copy_constructions == 0
    # assert cstats.move_constructions >= 0 # Doesn't invoke any
    assert cstats.copy_assignments == 0
    assert cstats.move_assignments == 0

    # Object
    cstats = ConstructorStats.get(m.Object)
    assert cstats.alive() == 0
    assert cstats.values() == []
    assert cstats.default_constructions == 10
    assert cstats.copy_constructions == 0
    # assert cstats.move_constructions >= 0 # Doesn't invoke any
    assert cstats.copy_assignments == 0
    assert cstats.move_assignments == 0

    # ref<>
    cstats = m.cstats_ref()
    assert cstats.alive() == 0
    assert cstats.values() == ['from pointer'] * 10
    assert cstats.default_constructions == 30
    assert cstats.copy_constructions == 12
    # assert cstats.move_constructions >= 0 # Doesn't invoke any
    assert cstats.copy_assignments == 30
    assert cstats.move_assignments == 0


def test_smart_ptr_refcounting():
    assert m.test_object1_refcounting()


def test_unique_nodelete():
    o = m.MyObject4(23)
    assert o.value == 23
    cstats = ConstructorStats.get(m.MyObject4)
    assert cstats.alive() == 1
    del o
    assert cstats.alive() == 1  # Leak, but that's intentional


def test_unique_nodelete4a():
    o = m.MyObject4a(23)
    assert o.value == 23
    cstats = ConstructorStats.get(m.MyObject4a)
    assert cstats.alive() == 1
    del o
    assert cstats.alive() == 1  # Leak, but that's intentional


def test_unique_deleter():
    o = m.MyObject4b(23)
    assert o.value == 23
    cstats4a = ConstructorStats.get(m.MyObject4a)
    assert cstats4a.alive() == 2  # Two becaue of previous test
    cstats4b = ConstructorStats.get(m.MyObject4b)
    assert cstats4b.alive() == 1
    del o
    assert cstats4a.alive() == 1  # Should now only be one leftover from previous test
    assert cstats4b.alive() == 0  # Should be deleted


def test_large_holder():
    o = m.MyObject5(5)
    assert o.value == 5
    cstats = ConstructorStats.get(m.MyObject5)
    assert cstats.alive() == 1
    del o
    assert cstats.alive() == 0


def test_shared_ptr_and_references():
    s = m.SharedPtrRef()
    stats = ConstructorStats.get(m.A)
    assert stats.alive() == 2

    ref = s.ref  # init_holder_helper(holder_ptr=false, owned=false)
    assert stats.alive() == 2
    assert s.set_ref(ref)
    with pytest.raises(RuntimeError) as excinfo:
        assert s.set_holder(ref)
    assert "Unable to cast from non-held to held instance" in str(excinfo.value)

    copy = s.copy  # init_holder_helper(holder_ptr=false, owned=true)
    assert stats.alive() == 3
    assert s.set_ref(copy)
    assert s.set_holder(copy)

    holder_ref = s.holder_ref  # init_holder_helper(holder_ptr=true, owned=false)
    assert stats.alive() == 3
    assert s.set_ref(holder_ref)
    assert s.set_holder(holder_ref)

    holder_copy = s.holder_copy  # init_holder_helper(holder_ptr=true, owned=true)
    assert stats.alive() == 3
    assert s.set_ref(holder_copy)
    assert s.set_holder(holder_copy)

    del ref, copy, holder_ref, holder_copy, s
    assert stats.alive() == 0


def test_shared_ptr_from_this_and_references():
    s = m.SharedFromThisRef()
    stats = ConstructorStats.get(m.B)
    assert stats.alive() == 2

    ref = s.ref  # init_holder_helper(holder_ptr=false, owned=false, bad_wp=false)
    assert stats.alive() == 2
    assert s.set_ref(ref)
    assert s.set_holder(ref)  # std::enable_shared_from_this can create a holder from a reference

    bad_wp = s.bad_wp  # init_holder_helper(holder_ptr=false, owned=false, bad_wp=true)
    assert stats.alive() == 2
    assert s.set_ref(bad_wp)
    with pytest.raises(RuntimeError) as excinfo:
        assert s.set_holder(bad_wp)
    assert "Unable to cast from non-held to held instance" in str(excinfo.value)

    copy = s.copy  # init_holder_helper(holder_ptr=false, owned=true, bad_wp=false)
    assert stats.alive() == 3
    assert s.set_ref(copy)
    assert s.set_holder(copy)

    holder_ref = s.holder_ref  # init_holder_helper(holder_ptr=true, owned=false, bad_wp=false)
    assert stats.alive() == 3
    assert s.set_ref(holder_ref)
    assert s.set_holder(holder_ref)

    holder_copy = s.holder_copy  # init_holder_helper(holder_ptr=true, owned=true, bad_wp=false)
    assert stats.alive() == 3
    assert s.set_ref(holder_copy)
    assert s.set_holder(holder_copy)

    del ref, bad_wp, copy, holder_ref, holder_copy, s
    assert stats.alive() == 0

    z = m.SharedFromThisVirt.get()
    y = m.SharedFromThisVirt.get()
    assert y is z


def test_move_only_holder():
    a = m.TypeWithMoveOnlyHolder.make()
    stats = ConstructorStats.get(m.TypeWithMoveOnlyHolder)
    assert stats.alive() == 1
    del a
    assert stats.alive() == 0


def test_holder_with_addressof_operator():
    # this test must not throw exception from c++
    a = m.TypeForHolderWithAddressOf.make()
    a.print_object_1()
    a.print_object_2()
    a.print_object_3()
    a.print_object_4()

    stats = ConstructorStats.get(m.TypeForHolderWithAddressOf)
    assert stats.alive() == 1

    np = m.TypeForHolderWithAddressOf.make()
    assert stats.alive() == 2
    del a
    assert stats.alive() == 1
    del np
    assert stats.alive() == 0

    b = m.TypeForHolderWithAddressOf.make()
    c = b
    assert b.get() is c.get()
    assert stats.alive() == 1

    del b
    assert stats.alive() == 1

    del c
    assert stats.alive() == 0


def test_move_only_holder_with_addressof_operator():
    a = m.TypeForMoveOnlyHolderWithAddressOf.make()
    a.print_object()

    stats = ConstructorStats.get(m.TypeForMoveOnlyHolderWithAddressOf)
    assert stats.alive() == 1

    a.value = 42
    assert a.value == 42

    del a
    assert stats.alive() == 0


def test_smart_ptr_from_default():
    instance = m.HeldByDefaultHolder()
    with pytest.raises(RuntimeError) as excinfo:
        m.HeldByDefaultHolder.load_shared_ptr(instance)
    assert "Unable to load a custom holder type from a default-holder instance" in str(excinfo)


def test_shared_ptr_gc():
    """#187: issue involving std::shared_ptr<> return value policy & garbage collection"""
    el = m.ElementList()
    for i in range(10):
        el.add(m.ElementA(i))
    pytest.gc_collect()
    for i, v in enumerate(el.get()):
        assert i == v.value()
