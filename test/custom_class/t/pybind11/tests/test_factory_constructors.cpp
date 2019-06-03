/*
    tests/test_factory_constructors.cpp -- tests construction from a factory function
                                           via py::init_factory()

    Copyright (c) 2017 Jason Rhinelander <jason@imaginary.ca>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include "constructor_stats.h"
#include <cmath>

// Classes for testing python construction via C++ factory function:
// Not publicly constructible, copyable, or movable:
class TestFactory1 {
    friend class TestFactoryHelper;
    TestFactory1() : value("(empty)") { print_default_created(this); }
    TestFactory1(int v) : value(std::to_string(v)) { print_created(this, value); }
    TestFactory1(std::string v) : value(std::move(v)) { print_created(this, value); }
    TestFactory1(TestFactory1 &&) = delete;
    TestFactory1(const TestFactory1 &) = delete;
    TestFactory1 &operator=(TestFactory1 &&) = delete;
    TestFactory1 &operator=(const TestFactory1 &) = delete;
public:
    std::string value;
    ~TestFactory1() { print_destroyed(this); }
};
// Non-public construction, but moveable:
class TestFactory2 {
    friend class TestFactoryHelper;
    TestFactory2() : value("(empty2)") { print_default_created(this); }
    TestFactory2(int v) : value(std::to_string(v)) { print_created(this, value); }
    TestFactory2(std::string v) : value(std::move(v)) { print_created(this, value); }
public:
    TestFactory2(TestFactory2 &&m) { value = std::move(m.value); print_move_created(this); }
    TestFactory2 &operator=(TestFactory2 &&m) { value = std::move(m.value); print_move_assigned(this); return *this; }
    std::string value;
    ~TestFactory2() { print_destroyed(this); }
};
// Mixed direct/factory construction:
class TestFactory3 {
protected:
    friend class TestFactoryHelper;
    TestFactory3() : value("(empty3)") { print_default_created(this); }
    TestFactory3(int v) : value(std::to_string(v)) { print_created(this, value); }
public:
    TestFactory3(std::string v) : value(std::move(v)) { print_created(this, value); }
    TestFactory3(TestFactory3 &&m) { value = std::move(m.value); print_move_created(this); }
    TestFactory3 &operator=(TestFactory3 &&m) { value = std::move(m.value); print_move_assigned(this); return *this; }
    std::string value;
    virtual ~TestFactory3() { print_destroyed(this); }
};
// Inheritance test
class TestFactory4 : public TestFactory3 {
public:
    TestFactory4() : TestFactory3() { print_default_created(this); }
    TestFactory4(int v) : TestFactory3(v) { print_created(this, v); }
    virtual ~TestFactory4() { print_destroyed(this); }
};
// Another class for an invalid downcast test
class TestFactory5 : public TestFactory3 {
public:
    TestFactory5(int i) : TestFactory3(i) { print_created(this, i); }
    virtual ~TestFactory5() { print_destroyed(this); }
};

class TestFactory6 {
protected:
    int value;
    bool alias = false;
public:
    TestFactory6(int i) : value{i} { print_created(this, i); }
    TestFactory6(TestFactory6 &&f) { print_move_created(this); value = f.value; alias = f.alias; }
    TestFactory6(const TestFactory6 &f) { print_copy_created(this); value = f.value; alias = f.alias; }
    virtual ~TestFactory6() { print_destroyed(this); }
    virtual int get() { return value; }
    bool has_alias() { return alias; }
};
class PyTF6 : public TestFactory6 {
public:
    // Special constructor that allows the factory to construct a PyTF6 from a TestFactory6 only
    // when an alias is needed:
    PyTF6(TestFactory6 &&base) : TestFactory6(std::move(base)) { alias = true; print_created(this, "move", value); }
    PyTF6(int i) : TestFactory6(i) { alias = true; print_created(this, i); }
    PyTF6(PyTF6 &&f) : TestFactory6(std::move(f)) { print_move_created(this); }
    PyTF6(const PyTF6 &f) : TestFactory6(f) { print_copy_created(this); }
    PyTF6(std::string s) : TestFactory6((int) s.size()) { alias = true; print_created(this, s); }
    virtual ~PyTF6() { print_destroyed(this); }
    int get() override { PYBIND11_OVERLOAD(int, TestFactory6, get, /*no args*/); }
};

class TestFactory7 {
protected:
    int value;
    bool alias = false;
public:
    TestFactory7(int i) : value{i} { print_created(this, i); }
    TestFactory7(TestFactory7 &&f) { print_move_created(this); value = f.value; alias = f.alias; }
    TestFactory7(const TestFactory7 &f) { print_copy_created(this); value = f.value; alias = f.alias; }
    virtual ~TestFactory7() { print_destroyed(this); }
    virtual int get() { return value; }
    bool has_alias() { return alias; }
};
class PyTF7 : public TestFactory7 {
public:
    PyTF7(int i) : TestFactory7(i) { alias = true; print_created(this, i); }
    PyTF7(PyTF7 &&f) : TestFactory7(std::move(f)) { print_move_created(this); }
    PyTF7(const PyTF7 &f) : TestFactory7(f) { print_copy_created(this); }
    virtual ~PyTF7() { print_destroyed(this); }
    int get() override { PYBIND11_OVERLOAD(int, TestFactory7, get, /*no args*/); }
};


class TestFactoryHelper {
public:
    // Non-movable, non-copyable type:
    // Return via pointer:
    static TestFactory1 *construct1() { return new TestFactory1(); }
    // Holder:
    static std::unique_ptr<TestFactory1> construct1(int a) { return std::unique_ptr<TestFactory1>(new TestFactory1(a)); }
    // pointer again
    static TestFactory1 *construct1_string(std::string a) { return new TestFactory1(a); }

    // Moveable type:
    // pointer:
    static TestFactory2 *construct2() { return new TestFactory2(); }
    // holder:
    static std::unique_ptr<TestFactory2> construct2(int a) { return std::unique_ptr<TestFactory2>(new TestFactory2(a)); }
    // by value moving:
    static TestFactory2 construct2(std::string a) { return TestFactory2(a); }

    // shared_ptr holder type:
    // pointer:
    static TestFactory3 *construct3() { return new TestFactory3(); }
    // holder:
    static std::shared_ptr<TestFactory3> construct3(int a) { return std::shared_ptr<TestFactory3>(new TestFactory3(a)); }
};

TEST_SUBMODULE(factory_constructors, m) {

    // Define various trivial types to allow simpler overload resolution:
    py::module m_tag = m.def_submodule("tag");
#define MAKE_TAG_TYPE(Name) \
    struct Name##_tag {}; \
    py::class_<Name##_tag>(m_tag, #Name "_tag").def(py::init<>()); \
    m_tag.attr(#Name) = py::cast(Name##_tag{})
    MAKE_TAG_TYPE(pointer);
    MAKE_TAG_TYPE(unique_ptr);
    MAKE_TAG_TYPE(move);
    MAKE_TAG_TYPE(shared_ptr);
    MAKE_TAG_TYPE(derived);
    MAKE_TAG_TYPE(TF4);
    MAKE_TAG_TYPE(TF5);
    MAKE_TAG_TYPE(null_ptr);
    MAKE_TAG_TYPE(base);
    MAKE_TAG_TYPE(invalid_base);
    MAKE_TAG_TYPE(alias);
    MAKE_TAG_TYPE(unaliasable);
    MAKE_TAG_TYPE(mixed);

    // test_init_factory_basic, test_bad_type
    py::class_<TestFactory1>(m, "TestFactory1")
        .def(py::init([](unique_ptr_tag, int v) { return TestFactoryHelper::construct1(v); }))
        .def(py::init(&TestFactoryHelper::construct1_string)) // raw function pointer
        .def(py::init([](pointer_tag) { return TestFactoryHelper::construct1(); }))
        .def(py::init([](py::handle, int v, py::handle) { return TestFactoryHelper::construct1(v); }))
        .def_readwrite("value", &TestFactory1::value)
        ;
    py::class_<TestFactory2>(m, "TestFactory2")
        .def(py::init([](pointer_tag, int v) { return TestFactoryHelper::construct2(v); }))
        .def(py::init([](unique_ptr_tag, std::string v) { return TestFactoryHelper::construct2(v); }))
        .def(py::init([](move_tag) { return TestFactoryHelper::construct2(); }))
        .def_readwrite("value", &TestFactory2::value)
        ;

    // Stateful & reused:
    int c = 1;
    auto c4a = [c](pointer_tag, TF4_tag, int a) { (void) c; return new TestFactory4(a);};

    // test_init_factory_basic, test_init_factory_casting
    py::class_<TestFactory3, std::shared_ptr<TestFactory3>>(m, "TestFactory3")
        .def(py::init([](pointer_tag, int v) { return TestFactoryHelper::construct3(v); }))
        .def(py::init([](shared_ptr_tag) { return TestFactoryHelper::construct3(); }))
        .def("__init__", [](TestFactory3 &self, std::string v) { new (&self) TestFactory3(v); }) // placement-new ctor

        // factories returning a derived type:
        .def(py::init(c4a)) // derived ptr
        .def(py::init([](pointer_tag, TF5_tag, int a) { return new TestFactory5(a); }))
        // derived shared ptr:
        .def(py::init([](shared_ptr_tag, TF4_tag, int a) { return std::make_shared<TestFactory4>(a); }))
        .def(py::init([](shared_ptr_tag, TF5_tag, int a) { return std::make_shared<TestFactory5>(a); }))

        // Returns nullptr:
        .def(py::init([](null_ptr_tag) { return (TestFactory3 *) nullptr; }))

        .def_readwrite("value", &TestFactory3::value)
        ;

    // test_init_factory_casting
    py::class_<TestFactory4, TestFactory3, std::shared_ptr<TestFactory4>>(m, "TestFactory4")
        .def(py::init(c4a)) // pointer
        ;

    // Doesn't need to be registered, but registering makes getting ConstructorStats easier:
    py::class_<TestFactory5, TestFactory3, std::shared_ptr<TestFactory5>>(m, "TestFactory5");

    // test_init_factory_alias
    // Alias testing
    py::class_<TestFactory6, PyTF6>(m, "TestFactory6")
        .def(py::init([](base_tag, int i) { return TestFactory6(i); }))
        .def(py::init([](alias_tag, int i) { return PyTF6(i); }))
        .def(py::init([](alias_tag, std::string s) { return PyTF6(s); }))
        .def(py::init([](alias_tag, pointer_tag, int i) { return new PyTF6(i); }))
        .def(py::init([](base_tag, pointer_tag, int i) { return new TestFactory6(i); }))
        .def(py::init([](base_tag, alias_tag, pointer_tag, int i) { return (TestFactory6 *) new PyTF6(i); }))

        .def("get", &TestFactory6::get)
        .def("has_alias", &TestFactory6::has_alias)

        .def_static("get_cstats", &ConstructorStats::get<TestFactory6>, py::return_value_policy::reference)
        .def_static("get_alias_cstats", &ConstructorStats::get<PyTF6>, py::return_value_policy::reference)
        ;

    // test_init_factory_dual
    // Separate alias constructor testing
    py::class_<TestFactory7, PyTF7, std::shared_ptr<TestFactory7>>(m, "TestFactory7")
        .def(py::init(
            [](int i) { return TestFactory7(i); },
            [](int i) { return PyTF7(i); }))
        .def(py::init(
            [](pointer_tag, int i) { return new TestFactory7(i); },
            [](pointer_tag, int i) { return new PyTF7(i); }))
        .def(py::init(
            [](mixed_tag, int i) { return new TestFactory7(i); },
            [](mixed_tag, int i) { return PyTF7(i); }))
        .def(py::init(
            [](mixed_tag, std::string s) { return TestFactory7((int) s.size()); },
            [](mixed_tag, std::string s) { return new PyTF7((int) s.size()); }))
        .def(py::init(
            [](base_tag, pointer_tag, int i) { return new TestFactory7(i); },
            [](base_tag, pointer_tag, int i) { return (TestFactory7 *) new PyTF7(i); }))
        .def(py::init(
            [](alias_tag, pointer_tag, int i) { return new PyTF7(i); },
            [](alias_tag, pointer_tag, int i) { return new PyTF7(10*i); }))
        .def(py::init(
            [](shared_ptr_tag, base_tag, int i) { return std::make_shared<TestFactory7>(i); },
            [](shared_ptr_tag, base_tag, int i) { auto *p = new PyTF7(i); return std::shared_ptr<TestFactory7>(p); }))
        .def(py::init(
            [](shared_ptr_tag, invalid_base_tag, int i) { return std::make_shared<TestFactory7>(i); },
            [](shared_ptr_tag, invalid_base_tag, int i) { return std::make_shared<TestFactory7>(i); })) // <-- invalid alias factory

        .def("get", &TestFactory7::get)
        .def("has_alias", &TestFactory7::has_alias)

        .def_static("get_cstats", &ConstructorStats::get<TestFactory7>, py::return_value_policy::reference)
        .def_static("get_alias_cstats", &ConstructorStats::get<PyTF7>, py::return_value_policy::reference)
        ;

    // test_placement_new_alternative
    // Class with a custom new operator but *without* a placement new operator (issue #948)
    class NoPlacementNew {
    public:
        NoPlacementNew(int i) : i(i) { }
        static void *operator new(std::size_t s) {
            auto *p = ::operator new(s);
            py::print("operator new called, returning", reinterpret_cast<uintptr_t>(p));
            return p;
        }
        static void operator delete(void *p) {
            py::print("operator delete called on", reinterpret_cast<uintptr_t>(p));
            ::operator delete(p);
        }
        int i;
    };
    // As of 2.2, `py::init<args>` no longer requires placement new
    py::class_<NoPlacementNew>(m, "NoPlacementNew")
        .def(py::init<int>())
        .def(py::init([]() { return new NoPlacementNew(100); }))
        .def_readwrite("i", &NoPlacementNew::i)
        ;


    // test_reallocations
    // Class that has verbose operator_new/operator_delete calls
    struct NoisyAlloc {
        NoisyAlloc(const NoisyAlloc &) = default;
        NoisyAlloc(int i) { py::print(py::str("NoisyAlloc(int {})").format(i)); }
        NoisyAlloc(double d) { py::print(py::str("NoisyAlloc(double {})").format(d)); }
        ~NoisyAlloc() { py::print("~NoisyAlloc()"); }

        static void *operator new(size_t s) { py::print("noisy new"); return ::operator new(s); }
        static void *operator new(size_t, void *p) { py::print("noisy placement new"); return p; }
        static void operator delete(void *p, size_t) { py::print("noisy delete"); ::operator delete(p); }
        static void operator delete(void *, void *) { py::print("noisy placement delete"); }
#if defined(_MSC_VER) && _MSC_VER < 1910
        // MSVC 2015 bug: the above "noisy delete" isn't invoked (fixed in MSVC 2017)
        static void operator delete(void *p) { py::print("noisy delete"); ::operator delete(p); }
#endif
    };
    py::class_<NoisyAlloc>(m, "NoisyAlloc")
        // Since these overloads have the same number of arguments, the dispatcher will try each of
        // them until the arguments convert.  Thus we can get a pre-allocation here when passing a
        // single non-integer:
        .def("__init__", [](NoisyAlloc *a, int i) { new (a) NoisyAlloc(i); }) // Regular constructor, runs first, requires preallocation
        .def(py::init([](double d) { return new NoisyAlloc(d); }))

        // The two-argument version: first the factory pointer overload.
        .def(py::init([](int i, int) { return new NoisyAlloc(i); }))
        // Return-by-value:
        .def(py::init([](double d, int) { return NoisyAlloc(d); }))
        // Old-style placement new init; requires preallocation
        .def("__init__", [](NoisyAlloc &a, double d, double) { new (&a) NoisyAlloc(d); })
        // Requires deallocation of previous overload preallocated value:
        .def(py::init([](int i, double) { return new NoisyAlloc(i); }))
        // Regular again: requires yet another preallocation
        .def("__init__", [](NoisyAlloc &a, int i, std::string) { new (&a) NoisyAlloc(i); })
        ;




    // static_assert testing (the following def's should all fail with appropriate compilation errors):
#if 0
    struct BadF1Base {};
    struct BadF1 : BadF1Base {};
    struct PyBadF1 : BadF1 {};
    py::class_<BadF1, PyBadF1, std::shared_ptr<BadF1>> bf1(m, "BadF1");
    // wrapped factory function must return a compatible pointer, holder, or value
    bf1.def(py::init([]() { return 3; }));
    // incompatible factory function pointer return type
    bf1.def(py::init([]() { static int three = 3; return &three; }));
    // incompatible factory function std::shared_ptr<T> return type: cannot convert shared_ptr<T> to holder
    // (non-polymorphic base)
    bf1.def(py::init([]() { return std::shared_ptr<BadF1Base>(new BadF1()); }));
#endif
}
