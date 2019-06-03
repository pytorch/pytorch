/*
    tests/test_smart_ptr.cpp -- binding classes with custom reference counting,
    implicit conversions between types

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#if defined(_MSC_VER) && _MSC_VER < 1910
#  pragma warning(disable: 4702) // unreachable code in system header
#endif

#include "pybind11_tests.h"
#include "object.h"

// Make pybind aware of the ref-counted wrapper type (s):

// ref<T> is a wrapper for 'Object' which uses intrusive reference counting
// It is always possible to construct a ref<T> from an Object* pointer without
// possible inconsistencies, hence the 'true' argument at the end.
PYBIND11_DECLARE_HOLDER_TYPE(T, ref<T>, true);
// Make pybind11 aware of the non-standard getter member function
namespace pybind11 { namespace detail {
    template <typename T>
    struct holder_helper<ref<T>> {
        static const T *get(const ref<T> &p) { return p.get_ptr(); }
    };
}}

// The following is not required anymore for std::shared_ptr, but it should compile without error:
PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

// This is just a wrapper around unique_ptr, but with extra fields to deliberately bloat up the
// holder size to trigger the non-simple-layout internal instance layout for single inheritance with
// large holder type:
template <typename T> class huge_unique_ptr {
    std::unique_ptr<T> ptr;
    uint64_t padding[10];
public:
    huge_unique_ptr(T *p) : ptr(p) {};
    T *get() { return ptr.get(); }
};
PYBIND11_DECLARE_HOLDER_TYPE(T, huge_unique_ptr<T>);

// Simple custom holder that works like unique_ptr
template <typename T>
class custom_unique_ptr {
    std::unique_ptr<T> impl;
public:
    custom_unique_ptr(T* p) : impl(p) { }
    T* get() const { return impl.get(); }
    T* release_ptr() { return impl.release(); }
};
PYBIND11_DECLARE_HOLDER_TYPE(T, custom_unique_ptr<T>);

// Simple custom holder that works like shared_ptr and has operator& overload
// To obtain address of an instance of this holder pybind should use std::addressof
// Attempt to get address via operator& may leads to segmentation fault
template <typename T>
class shared_ptr_with_addressof_operator {
    std::shared_ptr<T> impl;
public:
    shared_ptr_with_addressof_operator( ) = default;
    shared_ptr_with_addressof_operator(T* p) : impl(p) { }
    T* get() const { return impl.get(); }
    T** operator&() { throw std::logic_error("Call of overloaded operator& is not expected"); }
};
PYBIND11_DECLARE_HOLDER_TYPE(T, shared_ptr_with_addressof_operator<T>);

// Simple custom holder that works like unique_ptr and has operator& overload
// To obtain address of an instance of this holder pybind should use std::addressof
// Attempt to get address via operator& may leads to segmentation fault
template <typename T>
class unique_ptr_with_addressof_operator {
    std::unique_ptr<T> impl;
public:
    unique_ptr_with_addressof_operator() = default;
    unique_ptr_with_addressof_operator(T* p) : impl(p) { }
    T* get() const { return impl.get(); }
    T* release_ptr() { return impl.release(); }
    T** operator&() { throw std::logic_error("Call of overloaded operator& is not expected"); }
};
PYBIND11_DECLARE_HOLDER_TYPE(T, unique_ptr_with_addressof_operator<T>);


TEST_SUBMODULE(smart_ptr, m) {

    // test_smart_ptr

    // Object implementation in `object.h`
    py::class_<Object, ref<Object>> obj(m, "Object");
    obj.def("getRefCount", &Object::getRefCount);

    // Custom object with builtin reference counting (see 'object.h' for the implementation)
    class MyObject1 : public Object {
    public:
        MyObject1(int value) : value(value) { print_created(this, toString()); }
        std::string toString() const { return "MyObject1[" + std::to_string(value) + "]"; }
    protected:
        virtual ~MyObject1() { print_destroyed(this); }
    private:
        int value;
    };
    py::class_<MyObject1, ref<MyObject1>>(m, "MyObject1", obj)
        .def(py::init<int>());
    py::implicitly_convertible<py::int_, MyObject1>();

    m.def("make_object_1", []() -> Object * { return new MyObject1(1); });
    m.def("make_object_2", []() -> ref<Object> { return new MyObject1(2); });
    m.def("make_myobject1_1", []() -> MyObject1 * { return new MyObject1(4); });
    m.def("make_myobject1_2", []() -> ref<MyObject1> { return new MyObject1(5); });
    m.def("print_object_1", [](const Object *obj) { py::print(obj->toString()); });
    m.def("print_object_2", [](ref<Object> obj) { py::print(obj->toString()); });
    m.def("print_object_3", [](const ref<Object> &obj) { py::print(obj->toString()); });
    m.def("print_object_4", [](const ref<Object> *obj) { py::print((*obj)->toString()); });
    m.def("print_myobject1_1", [](const MyObject1 *obj) { py::print(obj->toString()); });
    m.def("print_myobject1_2", [](ref<MyObject1> obj) { py::print(obj->toString()); });
    m.def("print_myobject1_3", [](const ref<MyObject1> &obj) { py::print(obj->toString()); });
    m.def("print_myobject1_4", [](const ref<MyObject1> *obj) { py::print((*obj)->toString()); });

    // Expose constructor stats for the ref type
    m.def("cstats_ref", &ConstructorStats::get<ref_tag>);


    // Object managed by a std::shared_ptr<>
    class MyObject2 {
    public:
        MyObject2(const MyObject2 &) = default;
        MyObject2(int value) : value(value) { print_created(this, toString()); }
        std::string toString() const { return "MyObject2[" + std::to_string(value) + "]"; }
        virtual ~MyObject2() { print_destroyed(this); }
    private:
        int value;
    };
    py::class_<MyObject2, std::shared_ptr<MyObject2>>(m, "MyObject2")
        .def(py::init<int>());
    m.def("make_myobject2_1", []() { return new MyObject2(6); });
    m.def("make_myobject2_2", []() { return std::make_shared<MyObject2>(7); });
    m.def("print_myobject2_1", [](const MyObject2 *obj) { py::print(obj->toString()); });
    m.def("print_myobject2_2", [](std::shared_ptr<MyObject2> obj) { py::print(obj->toString()); });
    m.def("print_myobject2_3", [](const std::shared_ptr<MyObject2> &obj) { py::print(obj->toString()); });
    m.def("print_myobject2_4", [](const std::shared_ptr<MyObject2> *obj) { py::print((*obj)->toString()); });

    // Object managed by a std::shared_ptr<>, additionally derives from std::enable_shared_from_this<>
    class MyObject3 : public std::enable_shared_from_this<MyObject3> {
    public:
        MyObject3(const MyObject3 &) = default;
        MyObject3(int value) : value(value) { print_created(this, toString()); }
        std::string toString() const { return "MyObject3[" + std::to_string(value) + "]"; }
        virtual ~MyObject3() { print_destroyed(this); }
    private:
        int value;
    };
    py::class_<MyObject3, std::shared_ptr<MyObject3>>(m, "MyObject3")
        .def(py::init<int>());
    m.def("make_myobject3_1", []() { return new MyObject3(8); });
    m.def("make_myobject3_2", []() { return std::make_shared<MyObject3>(9); });
    m.def("print_myobject3_1", [](const MyObject3 *obj) { py::print(obj->toString()); });
    m.def("print_myobject3_2", [](std::shared_ptr<MyObject3> obj) { py::print(obj->toString()); });
    m.def("print_myobject3_3", [](const std::shared_ptr<MyObject3> &obj) { py::print(obj->toString()); });
    m.def("print_myobject3_4", [](const std::shared_ptr<MyObject3> *obj) { py::print((*obj)->toString()); });

    // test_smart_ptr_refcounting
    m.def("test_object1_refcounting", []() {
        ref<MyObject1> o = new MyObject1(0);
        bool good = o->getRefCount() == 1;
        py::object o2 = py::cast(o, py::return_value_policy::reference);
        // always request (partial) ownership for objects with intrusive
        // reference counting even when using the 'reference' RVP
        good &= o->getRefCount() == 2;
        return good;
    });

    // test_unique_nodelete
    // Object with a private destructor
    class MyObject4 {
    public:
        MyObject4(int value) : value{value} { print_created(this); }
        int value;
    private:
        ~MyObject4() { print_destroyed(this); }
    };
    py::class_<MyObject4, std::unique_ptr<MyObject4, py::nodelete>>(m, "MyObject4")
        .def(py::init<int>())
        .def_readwrite("value", &MyObject4::value);

    // test_unique_deleter
    // Object with std::unique_ptr<T, D> where D is not matching the base class
    // Object with a protected destructor
    class MyObject4a {
    public:
        MyObject4a(int i) {
            value = i;
            print_created(this);
        };
        int value;
    protected:
        virtual ~MyObject4a() { print_destroyed(this); }
    };
    py::class_<MyObject4a, std::unique_ptr<MyObject4a, py::nodelete>>(m, "MyObject4a")
        .def(py::init<int>())
        .def_readwrite("value", &MyObject4a::value);

    // Object derived but with public destructor and no Deleter in default holder
    class MyObject4b : public MyObject4a {
    public:
        MyObject4b(int i) : MyObject4a(i) { print_created(this); }
        ~MyObject4b() { print_destroyed(this); }
    };
    py::class_<MyObject4b, MyObject4a>(m, "MyObject4b")
        .def(py::init<int>());

    // test_large_holder
    class MyObject5 { // managed by huge_unique_ptr
    public:
        MyObject5(int value) : value{value} { print_created(this); }
        ~MyObject5() { print_destroyed(this); }
        int value;
    };
    py::class_<MyObject5, huge_unique_ptr<MyObject5>>(m, "MyObject5")
        .def(py::init<int>())
        .def_readwrite("value", &MyObject5::value);

    // test_shared_ptr_and_references
    struct SharedPtrRef {
        struct A {
            A() { print_created(this); }
            A(const A &) { print_copy_created(this); }
            A(A &&) { print_move_created(this); }
            ~A() { print_destroyed(this); }
        };

        A value = {};
        std::shared_ptr<A> shared = std::make_shared<A>();
    };
    using A = SharedPtrRef::A;
    py::class_<A, std::shared_ptr<A>>(m, "A");
    py::class_<SharedPtrRef>(m, "SharedPtrRef")
        .def(py::init<>())
        .def_readonly("ref", &SharedPtrRef::value)
        .def_property_readonly("copy", [](const SharedPtrRef &s) { return s.value; },
                               py::return_value_policy::copy)
        .def_readonly("holder_ref", &SharedPtrRef::shared)
        .def_property_readonly("holder_copy", [](const SharedPtrRef &s) { return s.shared; },
                               py::return_value_policy::copy)
        .def("set_ref", [](SharedPtrRef &, const A &) { return true; })
        .def("set_holder", [](SharedPtrRef &, std::shared_ptr<A>) { return true; });

    // test_shared_ptr_from_this_and_references
    struct SharedFromThisRef {
        struct B : std::enable_shared_from_this<B> {
            B() { print_created(this); }
            B(const B &) : std::enable_shared_from_this<B>() { print_copy_created(this); }
            B(B &&) : std::enable_shared_from_this<B>() { print_move_created(this); }
            ~B() { print_destroyed(this); }
        };

        B value = {};
        std::shared_ptr<B> shared = std::make_shared<B>();
    };
    using B = SharedFromThisRef::B;
    py::class_<B, std::shared_ptr<B>>(m, "B");
    py::class_<SharedFromThisRef>(m, "SharedFromThisRef")
        .def(py::init<>())
        .def_readonly("bad_wp", &SharedFromThisRef::value)
        .def_property_readonly("ref", [](const SharedFromThisRef &s) -> const B & { return *s.shared; })
        .def_property_readonly("copy", [](const SharedFromThisRef &s) { return s.value; },
                               py::return_value_policy::copy)
        .def_readonly("holder_ref", &SharedFromThisRef::shared)
        .def_property_readonly("holder_copy", [](const SharedFromThisRef &s) { return s.shared; },
                               py::return_value_policy::copy)
        .def("set_ref", [](SharedFromThisRef &, const B &) { return true; })
        .def("set_holder", [](SharedFromThisRef &, std::shared_ptr<B>) { return true; });

    // Issue #865: shared_from_this doesn't work with virtual inheritance
    struct SharedFromThisVBase : std::enable_shared_from_this<SharedFromThisVBase> {
        SharedFromThisVBase() = default;
        SharedFromThisVBase(const SharedFromThisVBase &) = default;
        virtual ~SharedFromThisVBase() = default;
    };
    struct SharedFromThisVirt : virtual SharedFromThisVBase {};
    static std::shared_ptr<SharedFromThisVirt> sft(new SharedFromThisVirt());
    py::class_<SharedFromThisVirt, std::shared_ptr<SharedFromThisVirt>>(m, "SharedFromThisVirt")
        .def_static("get", []() { return sft.get(); });

    // test_move_only_holder
    struct C {
        C() { print_created(this); }
        ~C() { print_destroyed(this); }
    };
    py::class_<C, custom_unique_ptr<C>>(m, "TypeWithMoveOnlyHolder")
        .def_static("make", []() { return custom_unique_ptr<C>(new C); });

    // test_holder_with_addressof_operator
    struct TypeForHolderWithAddressOf {
        TypeForHolderWithAddressOf() { print_created(this); }
        TypeForHolderWithAddressOf(const TypeForHolderWithAddressOf &) { print_copy_created(this); }
        TypeForHolderWithAddressOf(TypeForHolderWithAddressOf &&) { print_move_created(this); }
        ~TypeForHolderWithAddressOf() { print_destroyed(this); }
        std::string toString() const {
            return "TypeForHolderWithAddressOf[" + std::to_string(value) + "]";
        }
        int value = 42;
    };
    using HolderWithAddressOf = shared_ptr_with_addressof_operator<TypeForHolderWithAddressOf>;
    py::class_<TypeForHolderWithAddressOf, HolderWithAddressOf>(m, "TypeForHolderWithAddressOf")
        .def_static("make", []() { return HolderWithAddressOf(new TypeForHolderWithAddressOf); })
        .def("get", [](const HolderWithAddressOf &self) { return self.get(); })
        .def("print_object_1", [](const TypeForHolderWithAddressOf *obj) { py::print(obj->toString()); })
        .def("print_object_2", [](HolderWithAddressOf obj) { py::print(obj.get()->toString()); })
        .def("print_object_3", [](const HolderWithAddressOf &obj) { py::print(obj.get()->toString()); })
        .def("print_object_4", [](const HolderWithAddressOf *obj) { py::print((*obj).get()->toString()); });

    // test_move_only_holder_with_addressof_operator
    struct TypeForMoveOnlyHolderWithAddressOf {
        TypeForMoveOnlyHolderWithAddressOf(int value) : value{value} { print_created(this); }
        ~TypeForMoveOnlyHolderWithAddressOf() { print_destroyed(this); }
        std::string toString() const {
            return "MoveOnlyHolderWithAddressOf[" + std::to_string(value) + "]";
        }
        int value;
    };
    using MoveOnlyHolderWithAddressOf = unique_ptr_with_addressof_operator<TypeForMoveOnlyHolderWithAddressOf>;
    py::class_<TypeForMoveOnlyHolderWithAddressOf, MoveOnlyHolderWithAddressOf>(m, "TypeForMoveOnlyHolderWithAddressOf")
        .def_static("make", []() { return MoveOnlyHolderWithAddressOf(new TypeForMoveOnlyHolderWithAddressOf(0)); })
        .def_readwrite("value", &TypeForMoveOnlyHolderWithAddressOf::value)
        .def("print_object", [](const TypeForMoveOnlyHolderWithAddressOf *obj) { py::print(obj->toString()); });

    // test_smart_ptr_from_default
    struct HeldByDefaultHolder { };
    py::class_<HeldByDefaultHolder>(m, "HeldByDefaultHolder")
        .def(py::init<>())
        .def_static("load_shared_ptr", [](std::shared_ptr<HeldByDefaultHolder>) {});

    // test_shared_ptr_gc
    // #187: issue involving std::shared_ptr<> return value policy & garbage collection
    struct ElementBase { virtual void foo() { } /* Force creation of virtual table */ };
    py::class_<ElementBase, std::shared_ptr<ElementBase>>(m, "ElementBase");

    struct ElementA : ElementBase {
        ElementA(int v) : v(v) { }
        int value() { return v; }
        int v;
    };
    py::class_<ElementA, ElementBase, std::shared_ptr<ElementA>>(m, "ElementA")
        .def(py::init<int>())
        .def("value", &ElementA::value);

    struct ElementList {
        void add(std::shared_ptr<ElementBase> e) { l.push_back(e); }
        std::vector<std::shared_ptr<ElementBase>> l;
    };
    py::class_<ElementList, std::shared_ptr<ElementList>>(m, "ElementList")
        .def(py::init<>())
        .def("add", &ElementList::add)
        .def("get", [](ElementList &el) {
            py::list list;
            for (auto &e : el.l)
                list.append(py::cast(e));
            return list;
        });
}
