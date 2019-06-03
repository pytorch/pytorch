/*
    tests/test_methods_and_attributes.cpp -- constructors, deconstructors, attribute access,
    __str__, argument and return value conventions

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include "constructor_stats.h"

class ExampleMandA {
public:
    ExampleMandA() { print_default_created(this); }
    ExampleMandA(int value) : value(value) { print_created(this, value); }
    ExampleMandA(const ExampleMandA &e) : value(e.value) { print_copy_created(this); }
    ExampleMandA(ExampleMandA &&e) : value(e.value) { print_move_created(this); }
    ~ExampleMandA() { print_destroyed(this); }

    std::string toString() {
        return "ExampleMandA[value=" + std::to_string(value) + "]";
    }

    void operator=(const ExampleMandA &e) { print_copy_assigned(this); value = e.value; }
    void operator=(ExampleMandA &&e) { print_move_assigned(this); value = e.value; }

    void add1(ExampleMandA other) { value += other.value; }         // passing by value
    void add2(ExampleMandA &other) { value += other.value; }        // passing by reference
    void add3(const ExampleMandA &other) { value += other.value; }  // passing by const reference
    void add4(ExampleMandA *other) { value += other->value; }       // passing by pointer
    void add5(const ExampleMandA *other) { value += other->value; } // passing by const pointer

    void add6(int other) { value += other; }                        // passing by value
    void add7(int &other) { value += other; }                       // passing by reference
    void add8(const int &other) { value += other; }                 // passing by const reference
    void add9(int *other) { value += *other; }                      // passing by pointer
    void add10(const int *other) { value += *other; }               // passing by const pointer

    ExampleMandA self1() { return *this; }                          // return by value
    ExampleMandA &self2() { return *this; }                         // return by reference
    const ExampleMandA &self3() { return *this; }                   // return by const reference
    ExampleMandA *self4() { return this; }                          // return by pointer
    const ExampleMandA *self5() { return this; }                    // return by const pointer

    int internal1() { return value; }                               // return by value
    int &internal2() { return value; }                              // return by reference
    const int &internal3() { return value; }                        // return by const reference
    int *internal4() { return &value; }                             // return by pointer
    const int *internal5() { return &value; }                       // return by const pointer

    py::str overloaded()             { return "()"; }
    py::str overloaded(int)          { return "(int)"; }
    py::str overloaded(int, float)   { return "(int, float)"; }
    py::str overloaded(float, int)   { return "(float, int)"; }
    py::str overloaded(int, int)     { return "(int, int)"; }
    py::str overloaded(float, float) { return "(float, float)"; }
    py::str overloaded(int)          const { return "(int) const"; }
    py::str overloaded(int, float)   const { return "(int, float) const"; }
    py::str overloaded(float, int)   const { return "(float, int) const"; }
    py::str overloaded(int, int)     const { return "(int, int) const"; }
    py::str overloaded(float, float) const { return "(float, float) const"; }

    static py::str overloaded(float) { return "static float"; }

    int value = 0;
};

struct TestProperties {
    int value = 1;
    static int static_value;

    int get() const { return value; }
    void set(int v) { value = v; }

    static int static_get() { return static_value; }
    static void static_set(int v) { static_value = v; }
};
int TestProperties::static_value = 1;

struct TestPropertiesOverride : TestProperties {
    int value = 99;
    static int static_value;
};
int TestPropertiesOverride::static_value = 99;

struct TestPropRVP {
    UserType v1{1};
    UserType v2{1};
    static UserType sv1;
    static UserType sv2;

    const UserType &get1() const { return v1; }
    const UserType &get2() const { return v2; }
    UserType get_rvalue() const { return v2; }
    void set1(int v) { v1.set(v); }
    void set2(int v) { v2.set(v); }
};
UserType TestPropRVP::sv1(1);
UserType TestPropRVP::sv2(1);

// py::arg/py::arg_v testing: these arguments just record their argument when invoked
class ArgInspector1 { public: std::string arg = "(default arg inspector 1)"; };
class ArgInspector2 { public: std::string arg = "(default arg inspector 2)"; };
class ArgAlwaysConverts { };
namespace pybind11 { namespace detail {
template <> struct type_caster<ArgInspector1> {
public:
    PYBIND11_TYPE_CASTER(ArgInspector1, _("ArgInspector1"));

    bool load(handle src, bool convert) {
        value.arg = "loading ArgInspector1 argument " +
            std::string(convert ? "WITH" : "WITHOUT") + " conversion allowed.  "
            "Argument value = " + (std::string) str(src);
        return true;
    }

    static handle cast(const ArgInspector1 &src, return_value_policy, handle) {
        return str(src.arg).release();
    }
};
template <> struct type_caster<ArgInspector2> {
public:
    PYBIND11_TYPE_CASTER(ArgInspector2, _("ArgInspector2"));

    bool load(handle src, bool convert) {
        value.arg = "loading ArgInspector2 argument " +
            std::string(convert ? "WITH" : "WITHOUT") + " conversion allowed.  "
            "Argument value = " + (std::string) str(src);
        return true;
    }

    static handle cast(const ArgInspector2 &src, return_value_policy, handle) {
        return str(src.arg).release();
    }
};
template <> struct type_caster<ArgAlwaysConverts> {
public:
    PYBIND11_TYPE_CASTER(ArgAlwaysConverts, _("ArgAlwaysConverts"));

    bool load(handle, bool convert) {
        return convert;
    }

    static handle cast(const ArgAlwaysConverts &, return_value_policy, handle) {
        return py::none().release();
    }
};
}}

// test_custom_caster_destruction
class DestructionTester {
public:
    DestructionTester() { print_default_created(this); }
    ~DestructionTester() { print_destroyed(this); }
    DestructionTester(const DestructionTester &) { print_copy_created(this); }
    DestructionTester(DestructionTester &&) { print_move_created(this); }
    DestructionTester &operator=(const DestructionTester &) { print_copy_assigned(this); return *this; }
    DestructionTester &operator=(DestructionTester &&) { print_move_assigned(this); return *this; }
};
namespace pybind11 { namespace detail {
template <> struct type_caster<DestructionTester> {
    PYBIND11_TYPE_CASTER(DestructionTester, _("DestructionTester"));
    bool load(handle, bool) { return true; }

    static handle cast(const DestructionTester &, return_value_policy, handle) {
        return py::bool_(true).release();
    }
};
}}

// Test None-allowed py::arg argument policy
class NoneTester { public: int answer = 42; };
int none1(const NoneTester &obj) { return obj.answer; }
int none2(NoneTester *obj) { return obj ? obj->answer : -1; }
int none3(std::shared_ptr<NoneTester> &obj) { return obj ? obj->answer : -1; }
int none4(std::shared_ptr<NoneTester> *obj) { return obj && *obj ? (*obj)->answer : -1; }
int none5(std::shared_ptr<NoneTester> obj) { return obj ? obj->answer : -1; }

struct StrIssue {
    int val = -1;

    StrIssue() = default;
    StrIssue(int i) : val{i} {}
};

// Issues #854, #910: incompatible function args when member function/pointer is in unregistered base class
class UnregisteredBase {
public:
    void do_nothing() const {}
    void increase_value() { rw_value++; ro_value += 0.25; }
    void set_int(int v) { rw_value = v; }
    int get_int() const { return rw_value; }
    double get_double() const { return ro_value; }
    int rw_value = 42;
    double ro_value = 1.25;
};
class RegisteredDerived : public UnregisteredBase {
public:
    using UnregisteredBase::UnregisteredBase;
    double sum() const { return rw_value + ro_value; }
};

TEST_SUBMODULE(methods_and_attributes, m) {
    // test_methods_and_attributes
    py::class_<ExampleMandA> emna(m, "ExampleMandA");
    emna.def(py::init<>())
        .def(py::init<int>())
        .def(py::init<const ExampleMandA&>())
        .def("add1", &ExampleMandA::add1)
        .def("add2", &ExampleMandA::add2)
        .def("add3", &ExampleMandA::add3)
        .def("add4", &ExampleMandA::add4)
        .def("add5", &ExampleMandA::add5)
        .def("add6", &ExampleMandA::add6)
        .def("add7", &ExampleMandA::add7)
        .def("add8", &ExampleMandA::add8)
        .def("add9", &ExampleMandA::add9)
        .def("add10", &ExampleMandA::add10)
        .def("self1", &ExampleMandA::self1)
        .def("self2", &ExampleMandA::self2)
        .def("self3", &ExampleMandA::self3)
        .def("self4", &ExampleMandA::self4)
        .def("self5", &ExampleMandA::self5)
        .def("internal1", &ExampleMandA::internal1)
        .def("internal2", &ExampleMandA::internal2)
        .def("internal3", &ExampleMandA::internal3)
        .def("internal4", &ExampleMandA::internal4)
        .def("internal5", &ExampleMandA::internal5)
#if defined(PYBIND11_OVERLOAD_CAST)
        .def("overloaded", py::overload_cast<>(&ExampleMandA::overloaded))
        .def("overloaded", py::overload_cast<int>(&ExampleMandA::overloaded))
        .def("overloaded", py::overload_cast<int,   float>(&ExampleMandA::overloaded))
        .def("overloaded", py::overload_cast<float,   int>(&ExampleMandA::overloaded))
        .def("overloaded", py::overload_cast<int,     int>(&ExampleMandA::overloaded))
        .def("overloaded", py::overload_cast<float, float>(&ExampleMandA::overloaded))
        .def("overloaded_float", py::overload_cast<float, float>(&ExampleMandA::overloaded))
        .def("overloaded_const", py::overload_cast<int         >(&ExampleMandA::overloaded, py::const_))
        .def("overloaded_const", py::overload_cast<int,   float>(&ExampleMandA::overloaded, py::const_))
        .def("overloaded_const", py::overload_cast<float,   int>(&ExampleMandA::overloaded, py::const_))
        .def("overloaded_const", py::overload_cast<int,     int>(&ExampleMandA::overloaded, py::const_))
        .def("overloaded_const", py::overload_cast<float, float>(&ExampleMandA::overloaded, py::const_))
#else
        .def("overloaded", static_cast<py::str (ExampleMandA::*)()>(&ExampleMandA::overloaded))
        .def("overloaded", static_cast<py::str (ExampleMandA::*)(int)>(&ExampleMandA::overloaded))
        .def("overloaded", static_cast<py::str (ExampleMandA::*)(int,   float)>(&ExampleMandA::overloaded))
        .def("overloaded", static_cast<py::str (ExampleMandA::*)(float,   int)>(&ExampleMandA::overloaded))
        .def("overloaded", static_cast<py::str (ExampleMandA::*)(int,     int)>(&ExampleMandA::overloaded))
        .def("overloaded", static_cast<py::str (ExampleMandA::*)(float, float)>(&ExampleMandA::overloaded))
        .def("overloaded_float", static_cast<py::str (ExampleMandA::*)(float, float)>(&ExampleMandA::overloaded))
        .def("overloaded_const", static_cast<py::str (ExampleMandA::*)(int         ) const>(&ExampleMandA::overloaded))
        .def("overloaded_const", static_cast<py::str (ExampleMandA::*)(int,   float) const>(&ExampleMandA::overloaded))
        .def("overloaded_const", static_cast<py::str (ExampleMandA::*)(float,   int) const>(&ExampleMandA::overloaded))
        .def("overloaded_const", static_cast<py::str (ExampleMandA::*)(int,     int) const>(&ExampleMandA::overloaded))
        .def("overloaded_const", static_cast<py::str (ExampleMandA::*)(float, float) const>(&ExampleMandA::overloaded))
#endif
        // test_no_mixed_overloads
        // Raise error if trying to mix static/non-static overloads on the same name:
        .def_static("add_mixed_overloads1", []() {
            auto emna = py::reinterpret_borrow<py::class_<ExampleMandA>>(py::module::import("pybind11_tests.methods_and_attributes").attr("ExampleMandA"));
            emna.def       ("overload_mixed1", static_cast<py::str (ExampleMandA::*)(int, int)>(&ExampleMandA::overloaded))
                .def_static("overload_mixed1", static_cast<py::str (              *)(float   )>(&ExampleMandA::overloaded));
        })
        .def_static("add_mixed_overloads2", []() {
            auto emna = py::reinterpret_borrow<py::class_<ExampleMandA>>(py::module::import("pybind11_tests.methods_and_attributes").attr("ExampleMandA"));
            emna.def_static("overload_mixed2", static_cast<py::str (              *)(float   )>(&ExampleMandA::overloaded))
                .def       ("overload_mixed2", static_cast<py::str (ExampleMandA::*)(int, int)>(&ExampleMandA::overloaded));
        })
        .def("__str__", &ExampleMandA::toString)
        .def_readwrite("value", &ExampleMandA::value);

    // test_copy_method
    // Issue #443: can't call copied methods in Python 3
    emna.attr("add2b") = emna.attr("add2");

    // test_properties, test_static_properties, test_static_cls
    py::class_<TestProperties>(m, "TestProperties")
        .def(py::init<>())
        .def_readonly("def_readonly", &TestProperties::value)
        .def_readwrite("def_readwrite", &TestProperties::value)
        .def_property("def_writeonly", nullptr,
                      [](TestProperties& s,int v) { s.value = v; } )
        .def_property("def_property_writeonly", nullptr, &TestProperties::set)
        .def_property_readonly("def_property_readonly", &TestProperties::get)
        .def_property("def_property", &TestProperties::get, &TestProperties::set)
        .def_property("def_property_impossible", nullptr, nullptr)
        .def_readonly_static("def_readonly_static", &TestProperties::static_value)
        .def_readwrite_static("def_readwrite_static", &TestProperties::static_value)
        .def_property_static("def_writeonly_static", nullptr,
                             [](py::object, int v) { TestProperties::static_value = v; })
        .def_property_readonly_static("def_property_readonly_static",
                                      [](py::object) { return TestProperties::static_get(); })
        .def_property_static("def_property_writeonly_static", nullptr,
                             [](py::object, int v) { return TestProperties::static_set(v); })
        .def_property_static("def_property_static",
                             [](py::object) { return TestProperties::static_get(); },
                             [](py::object, int v) { TestProperties::static_set(v); })
        .def_property_static("static_cls",
                             [](py::object cls) { return cls; },
                             [](py::object cls, py::function f) { f(cls); });

    py::class_<TestPropertiesOverride, TestProperties>(m, "TestPropertiesOverride")
        .def(py::init<>())
        .def_readonly("def_readonly", &TestPropertiesOverride::value)
        .def_readonly_static("def_readonly_static", &TestPropertiesOverride::static_value);

    auto static_get1 = [](py::object) -> const UserType & { return TestPropRVP::sv1; };
    auto static_get2 = [](py::object) -> const UserType & { return TestPropRVP::sv2; };
    auto static_set1 = [](py::object, int v) { TestPropRVP::sv1.set(v); };
    auto static_set2 = [](py::object, int v) { TestPropRVP::sv2.set(v); };
    auto rvp_copy = py::return_value_policy::copy;

    // test_property_return_value_policies
    py::class_<TestPropRVP>(m, "TestPropRVP")
        .def(py::init<>())
        .def_property_readonly("ro_ref", &TestPropRVP::get1)
        .def_property_readonly("ro_copy", &TestPropRVP::get2, rvp_copy)
        .def_property_readonly("ro_func", py::cpp_function(&TestPropRVP::get2, rvp_copy))
        .def_property("rw_ref", &TestPropRVP::get1, &TestPropRVP::set1)
        .def_property("rw_copy", &TestPropRVP::get2, &TestPropRVP::set2, rvp_copy)
        .def_property("rw_func", py::cpp_function(&TestPropRVP::get2, rvp_copy), &TestPropRVP::set2)
        .def_property_readonly_static("static_ro_ref", static_get1)
        .def_property_readonly_static("static_ro_copy", static_get2, rvp_copy)
        .def_property_readonly_static("static_ro_func", py::cpp_function(static_get2, rvp_copy))
        .def_property_static("static_rw_ref", static_get1, static_set1)
        .def_property_static("static_rw_copy", static_get2, static_set2, rvp_copy)
        .def_property_static("static_rw_func", py::cpp_function(static_get2, rvp_copy), static_set2)
        // test_property_rvalue_policy
        .def_property_readonly("rvalue", &TestPropRVP::get_rvalue)
        .def_property_readonly_static("static_rvalue", [](py::object) { return UserType(1); });

    // test_metaclass_override
    struct MetaclassOverride { };
    py::class_<MetaclassOverride>(m, "MetaclassOverride", py::metaclass((PyObject *) &PyType_Type))
        .def_property_readonly_static("readonly", [](py::object) { return 1; });

#if !defined(PYPY_VERSION)
    // test_dynamic_attributes
    class DynamicClass {
    public:
        DynamicClass() { print_default_created(this); }
        ~DynamicClass() { print_destroyed(this); }
    };
    py::class_<DynamicClass>(m, "DynamicClass", py::dynamic_attr())
        .def(py::init());

    class CppDerivedDynamicClass : public DynamicClass { };
    py::class_<CppDerivedDynamicClass, DynamicClass>(m, "CppDerivedDynamicClass")
        .def(py::init());
#endif

    // test_noconvert_args
    //
    // Test converting.  The ArgAlwaysConverts is just there to make the first no-conversion pass
    // fail so that our call always ends up happening via the second dispatch (the one that allows
    // some conversion).
    class ArgInspector {
    public:
        ArgInspector1 f(ArgInspector1 a, ArgAlwaysConverts) { return a; }
        std::string g(ArgInspector1 a, const ArgInspector1 &b, int c, ArgInspector2 *d, ArgAlwaysConverts) {
            return a.arg + "\n" + b.arg + "\n" + std::to_string(c) + "\n" + d->arg;
        }
        static ArgInspector2 h(ArgInspector2 a, ArgAlwaysConverts) { return a; }
    };
    py::class_<ArgInspector>(m, "ArgInspector")
        .def(py::init<>())
        .def("f", &ArgInspector::f, py::arg(), py::arg() = ArgAlwaysConverts())
        .def("g", &ArgInspector::g, "a"_a.noconvert(), "b"_a, "c"_a.noconvert()=13, "d"_a=ArgInspector2(), py::arg() = ArgAlwaysConverts())
        .def_static("h", &ArgInspector::h, py::arg().noconvert(), py::arg() = ArgAlwaysConverts())
        ;
    m.def("arg_inspect_func", [](ArgInspector2 a, ArgInspector1 b, ArgAlwaysConverts) { return a.arg + "\n" + b.arg; },
            py::arg().noconvert(false), py::arg_v(nullptr, ArgInspector1()).noconvert(true), py::arg() = ArgAlwaysConverts());

    m.def("floats_preferred", [](double f) { return 0.5 * f; }, py::arg("f"));
    m.def("floats_only", [](double f) { return 0.5 * f; }, py::arg("f").noconvert());
    m.def("ints_preferred", [](int i) { return i / 2; }, py::arg("i"));
    m.def("ints_only", [](int i) { return i / 2; }, py::arg("i").noconvert());

    // test_bad_arg_default
    // Issue/PR #648: bad arg default debugging output
#if !defined(NDEBUG)
    m.attr("debug_enabled") = true;
#else
    m.attr("debug_enabled") = false;
#endif
    m.def("bad_arg_def_named", []{
        auto m = py::module::import("pybind11_tests");
        m.def("should_fail", [](int, UnregisteredType) {}, py::arg(), py::arg("a") = UnregisteredType());
    });
    m.def("bad_arg_def_unnamed", []{
        auto m = py::module::import("pybind11_tests");
        m.def("should_fail", [](int, UnregisteredType) {}, py::arg(), py::arg() = UnregisteredType());
    });

    // test_accepts_none
    py::class_<NoneTester, std::shared_ptr<NoneTester>>(m, "NoneTester")
        .def(py::init<>());
    m.def("no_none1", &none1, py::arg().none(false));
    m.def("no_none2", &none2, py::arg().none(false));
    m.def("no_none3", &none3, py::arg().none(false));
    m.def("no_none4", &none4, py::arg().none(false));
    m.def("no_none5", &none5, py::arg().none(false));
    m.def("ok_none1", &none1);
    m.def("ok_none2", &none2, py::arg().none(true));
    m.def("ok_none3", &none3);
    m.def("ok_none4", &none4, py::arg().none(true));
    m.def("ok_none5", &none5);

    // test_str_issue
    // Issue #283: __str__ called on uninitialized instance when constructor arguments invalid
    py::class_<StrIssue>(m, "StrIssue")
        .def(py::init<int>())
        .def(py::init<>())
        .def("__str__", [](const StrIssue &si) {
            return "StrIssue[" + std::to_string(si.val) + "]"; }
        );

    // test_unregistered_base_implementations
    //
    // Issues #854/910: incompatible function args when member function/pointer is in unregistered
    // base class The methods and member pointers below actually resolve to members/pointers in
    // UnregisteredBase; before this test/fix they would be registered via lambda with a first
    // argument of an unregistered type, and thus uncallable.
    py::class_<RegisteredDerived>(m, "RegisteredDerived")
        .def(py::init<>())
        .def("do_nothing", &RegisteredDerived::do_nothing)
        .def("increase_value", &RegisteredDerived::increase_value)
        .def_readwrite("rw_value", &RegisteredDerived::rw_value)
        .def_readonly("ro_value", &RegisteredDerived::ro_value)
        // These should trigger a static_assert if uncommented
        //.def_readwrite("fails", &UserType::value) // should trigger a static_assert if uncommented
        //.def_readonly("fails", &UserType::value) // should trigger a static_assert if uncommented
        .def_property("rw_value_prop", &RegisteredDerived::get_int, &RegisteredDerived::set_int)
        .def_property_readonly("ro_value_prop", &RegisteredDerived::get_double)
        // This one is in the registered class:
        .def("sum", &RegisteredDerived::sum)
        ;

    using Adapted = decltype(py::method_adaptor<RegisteredDerived>(&RegisteredDerived::do_nothing));
    static_assert(std::is_same<Adapted, void (RegisteredDerived::*)() const>::value, "");

    // test_custom_caster_destruction
    // Test that `take_ownership` works on types with a custom type caster when given a pointer

    // default policy: don't take ownership:
    m.def("custom_caster_no_destroy", []() { static auto *dt = new DestructionTester(); return dt; });

    m.def("custom_caster_destroy", []() { return new DestructionTester(); },
            py::return_value_policy::take_ownership); // Takes ownership: destroy when finished
    m.def("custom_caster_destroy_const", []() -> const DestructionTester * { return new DestructionTester(); },
            py::return_value_policy::take_ownership); // Likewise (const doesn't inhibit destruction)
    m.def("destruction_tester_cstats", &ConstructorStats::get<DestructionTester>, py::return_value_policy::reference);
}
