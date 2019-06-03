/*
    tests/test_virtual_functions.cpp -- overriding virtual functions from Python

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include "constructor_stats.h"
#include <pybind11/functional.h>
#include <thread>

/* This is an example class that we'll want to be able to extend from Python */
class ExampleVirt  {
public:
    ExampleVirt(int state) : state(state) { print_created(this, state); }
    ExampleVirt(const ExampleVirt &e) : state(e.state) { print_copy_created(this); }
    ExampleVirt(ExampleVirt &&e) : state(e.state) { print_move_created(this); e.state = 0; }
    virtual ~ExampleVirt() { print_destroyed(this); }

    virtual int run(int value) {
        py::print("Original implementation of "
                  "ExampleVirt::run(state={}, value={}, str1={}, str2={})"_s.format(state, value, get_string1(), *get_string2()));
        return state + value;
    }

    virtual bool run_bool() = 0;
    virtual void pure_virtual() = 0;

    // Returning a reference/pointer to a type converted from python (numbers, strings, etc.) is a
    // bit trickier, because the actual int& or std::string& or whatever only exists temporarily, so
    // we have to handle it specially in the trampoline class (see below).
    virtual const std::string &get_string1() { return str1; }
    virtual const std::string *get_string2() { return &str2; }

private:
    int state;
    const std::string str1{"default1"}, str2{"default2"};
};

/* This is a wrapper class that must be generated */
class PyExampleVirt : public ExampleVirt {
public:
    using ExampleVirt::ExampleVirt; /* Inherit constructors */

    int run(int value) override {
        /* Generate wrapping code that enables native function overloading */
        PYBIND11_OVERLOAD(
            int,         /* Return type */
            ExampleVirt, /* Parent class */
            run,         /* Name of function */
            value        /* Argument(s) */
        );
    }

    bool run_bool() override {
        PYBIND11_OVERLOAD_PURE(
            bool,         /* Return type */
            ExampleVirt,  /* Parent class */
            run_bool,     /* Name of function */
                          /* This function has no arguments. The trailing comma
                             in the previous line is needed for some compilers */
        );
    }

    void pure_virtual() override {
        PYBIND11_OVERLOAD_PURE(
            void,         /* Return type */
            ExampleVirt,  /* Parent class */
            pure_virtual, /* Name of function */
                          /* This function has no arguments. The trailing comma
                             in the previous line is needed for some compilers */
        );
    }

    // We can return reference types for compatibility with C++ virtual interfaces that do so, but
    // note they have some significant limitations (see the documentation).
    const std::string &get_string1() override {
        PYBIND11_OVERLOAD(
            const std::string &, /* Return type */
            ExampleVirt,         /* Parent class */
            get_string1,         /* Name of function */
                                 /* (no arguments) */
        );
    }

    const std::string *get_string2() override {
        PYBIND11_OVERLOAD(
            const std::string *, /* Return type */
            ExampleVirt,         /* Parent class */
            get_string2,         /* Name of function */
                                 /* (no arguments) */
        );
    }

};

class NonCopyable {
public:
    NonCopyable(int a, int b) : value{new int(a*b)} { print_created(this, a, b); }
    NonCopyable(NonCopyable &&o) { value = std::move(o.value); print_move_created(this); }
    NonCopyable(const NonCopyable &) = delete;
    NonCopyable() = delete;
    void operator=(const NonCopyable &) = delete;
    void operator=(NonCopyable &&) = delete;
    std::string get_value() const {
        if (value) return std::to_string(*value); else return "(null)";
    }
    ~NonCopyable() { print_destroyed(this); }

private:
    std::unique_ptr<int> value;
};

// This is like the above, but is both copy and movable.  In effect this means it should get moved
// when it is not referenced elsewhere, but copied if it is still referenced.
class Movable {
public:
    Movable(int a, int b) : value{a+b} { print_created(this, a, b); }
    Movable(const Movable &m) { value = m.value; print_copy_created(this); }
    Movable(Movable &&m) { value = std::move(m.value); print_move_created(this); }
    std::string get_value() const { return std::to_string(value); }
    ~Movable() { print_destroyed(this); }
private:
    int value;
};

class NCVirt {
public:
    virtual NonCopyable get_noncopyable(int a, int b) { return NonCopyable(a, b); }
    virtual Movable get_movable(int a, int b) = 0;

    std::string print_nc(int a, int b) { return get_noncopyable(a, b).get_value(); }
    std::string print_movable(int a, int b) { return get_movable(a, b).get_value(); }
};
class NCVirtTrampoline : public NCVirt {
#if !defined(__INTEL_COMPILER)
    NonCopyable get_noncopyable(int a, int b) override {
        PYBIND11_OVERLOAD(NonCopyable, NCVirt, get_noncopyable, a, b);
    }
#endif
    Movable get_movable(int a, int b) override {
        PYBIND11_OVERLOAD_PURE(Movable, NCVirt, get_movable, a, b);
    }
};

struct Base {
    /* for some reason MSVC2015 can't compile this if the function is pure virtual */
    virtual std::string dispatch() const { return {}; };
    virtual ~Base() = default;
};

struct DispatchIssue : Base {
    virtual std::string dispatch() const {
        PYBIND11_OVERLOAD_PURE(std::string, Base, dispatch, /* no arguments */);
    }
};

static void test_gil() {
    {
        py::gil_scoped_acquire lock;
        py::print("1st lock acquired");

    }

    {
        py::gil_scoped_acquire lock;
        py::print("2nd lock acquired");
    }

}

static void test_gil_from_thread() {
    py::gil_scoped_release release;

    std::thread t(test_gil);
    t.join();
}


// Forward declaration (so that we can put the main tests here; the inherited virtual approaches are
// rather long).
void initialize_inherited_virtuals(py::module &m);

TEST_SUBMODULE(virtual_functions, m) {
    // test_override
    py::class_<ExampleVirt, PyExampleVirt>(m, "ExampleVirt")
        .def(py::init<int>())
        /* Reference original class in function definitions */
        .def("run", &ExampleVirt::run)
        .def("run_bool", &ExampleVirt::run_bool)
        .def("pure_virtual", &ExampleVirt::pure_virtual);

    py::class_<NonCopyable>(m, "NonCopyable")
        .def(py::init<int, int>());

    py::class_<Movable>(m, "Movable")
        .def(py::init<int, int>());

    // test_move_support
#if !defined(__INTEL_COMPILER)
    py::class_<NCVirt, NCVirtTrampoline>(m, "NCVirt")
        .def(py::init<>())
        .def("get_noncopyable", &NCVirt::get_noncopyable)
        .def("get_movable", &NCVirt::get_movable)
        .def("print_nc", &NCVirt::print_nc)
        .def("print_movable", &NCVirt::print_movable);
#endif

    m.def("runExampleVirt", [](ExampleVirt *ex, int value) { return ex->run(value); });
    m.def("runExampleVirtBool", [](ExampleVirt* ex) { return ex->run_bool(); });
    m.def("runExampleVirtVirtual", [](ExampleVirt *ex) { ex->pure_virtual(); });

    m.def("cstats_debug", &ConstructorStats::get<ExampleVirt>);
    initialize_inherited_virtuals(m);

    // test_alias_delay_initialization1
    // don't invoke Python dispatch classes by default when instantiating C++ classes
    // that were not extended on the Python side
    struct A {
        virtual ~A() {}
        virtual void f() { py::print("A.f()"); }
    };

    struct PyA : A {
        PyA() { py::print("PyA.PyA()"); }
        ~PyA() { py::print("PyA.~PyA()"); }

        void f() override {
            py::print("PyA.f()");
            // This convolution just gives a `void`, but tests that PYBIND11_TYPE() works to protect
            // a type containing a ,
            PYBIND11_OVERLOAD(PYBIND11_TYPE(typename std::enable_if<true, void>::type), A, f);
        }
    };

    py::class_<A, PyA>(m, "A")
        .def(py::init<>())
        .def("f", &A::f);

    m.def("call_f", [](A *a) { a->f(); });

    // test_alias_delay_initialization2
    // ... unless we explicitly request it, as in this example:
    struct A2 {
        virtual ~A2() {}
        virtual void f() { py::print("A2.f()"); }
    };

    struct PyA2 : A2 {
        PyA2() { py::print("PyA2.PyA2()"); }
        ~PyA2() { py::print("PyA2.~PyA2()"); }
        void f() override {
            py::print("PyA2.f()");
            PYBIND11_OVERLOAD(void, A2, f);
        }
    };

    py::class_<A2, PyA2>(m, "A2")
        .def(py::init_alias<>())
        .def(py::init([](int) { return new PyA2(); }))
        .def("f", &A2::f);

    m.def("call_f", [](A2 *a2) { a2->f(); });

    // test_dispatch_issue
    // #159: virtual function dispatch has problems with similar-named functions
    py::class_<Base, DispatchIssue>(m, "DispatchIssue")
        .def(py::init<>())
        .def("dispatch", &Base::dispatch);

    m.def("dispatch_issue_go", [](const Base * b) { return b->dispatch(); });

    // test_override_ref
    // #392/397: overriding reference-returning functions
    class OverrideTest {
    public:
        struct A { std::string value = "hi"; };
        std::string v;
        A a;
        explicit OverrideTest(const std::string &v) : v{v} {}
        virtual std::string str_value() { return v; }
        virtual std::string &str_ref() { return v; }
        virtual A A_value() { return a; }
        virtual A &A_ref() { return a; }
        virtual ~OverrideTest() = default;
    };

    class PyOverrideTest : public OverrideTest {
    public:
        using OverrideTest::OverrideTest;
        std::string str_value() override { PYBIND11_OVERLOAD(std::string, OverrideTest, str_value); }
        // Not allowed (uncommenting should hit a static_assert failure): we can't get a reference
        // to a python numeric value, since we only copy values in the numeric type caster:
//      std::string &str_ref() override { PYBIND11_OVERLOAD(std::string &, OverrideTest, str_ref); }
        // But we can work around it like this:
    private:
        std::string _tmp;
        std::string str_ref_helper() { PYBIND11_OVERLOAD(std::string, OverrideTest, str_ref); }
    public:
        std::string &str_ref() override { return _tmp = str_ref_helper(); }

        A A_value() override { PYBIND11_OVERLOAD(A, OverrideTest, A_value); }
        A &A_ref() override { PYBIND11_OVERLOAD(A &, OverrideTest, A_ref); }
    };

    py::class_<OverrideTest::A>(m, "OverrideTest_A")
        .def_readwrite("value", &OverrideTest::A::value);
    py::class_<OverrideTest, PyOverrideTest>(m, "OverrideTest")
        .def(py::init<const std::string &>())
        .def("str_value", &OverrideTest::str_value)
//      .def("str_ref", &OverrideTest::str_ref)
        .def("A_value", &OverrideTest::A_value)
        .def("A_ref", &OverrideTest::A_ref);
}


// Inheriting virtual methods.  We do two versions here: the repeat-everything version and the
// templated trampoline versions mentioned in docs/advanced.rst.
//
// These base classes are exactly the same, but we technically need distinct
// classes for this example code because we need to be able to bind them
// properly (pybind11, sensibly, doesn't allow us to bind the same C++ class to
// multiple python classes).
class A_Repeat {
#define A_METHODS \
public: \
    virtual int unlucky_number() = 0; \
    virtual std::string say_something(unsigned times) { \
        std::string s = ""; \
        for (unsigned i = 0; i < times; ++i) \
            s += "hi"; \
        return s; \
    } \
    std::string say_everything() { \
        return say_something(1) + " " + std::to_string(unlucky_number()); \
    }
A_METHODS
    virtual ~A_Repeat() = default;
};
class B_Repeat : public A_Repeat {
#define B_METHODS \
public: \
    int unlucky_number() override { return 13; } \
    std::string say_something(unsigned times) override { \
        return "B says hi " + std::to_string(times) + " times"; \
    } \
    virtual double lucky_number() { return 7.0; }
B_METHODS
};
class C_Repeat : public B_Repeat {
#define C_METHODS \
public: \
    int unlucky_number() override { return 4444; } \
    double lucky_number() override { return 888; }
C_METHODS
};
class D_Repeat : public C_Repeat {
#define D_METHODS // Nothing overridden.
D_METHODS
};

// Base classes for templated inheritance trampolines.  Identical to the repeat-everything version:
class A_Tpl { A_METHODS; virtual ~A_Tpl() = default; };
class B_Tpl : public A_Tpl { B_METHODS };
class C_Tpl : public B_Tpl { C_METHODS };
class D_Tpl : public C_Tpl { D_METHODS };


// Inheritance approach 1: each trampoline gets every virtual method (11 in total)
class PyA_Repeat : public A_Repeat {
public:
    using A_Repeat::A_Repeat;
    int unlucky_number() override { PYBIND11_OVERLOAD_PURE(int, A_Repeat, unlucky_number, ); }
    std::string say_something(unsigned times) override { PYBIND11_OVERLOAD(std::string, A_Repeat, say_something, times); }
};
class PyB_Repeat : public B_Repeat {
public:
    using B_Repeat::B_Repeat;
    int unlucky_number() override { PYBIND11_OVERLOAD(int, B_Repeat, unlucky_number, ); }
    std::string say_something(unsigned times) override { PYBIND11_OVERLOAD(std::string, B_Repeat, say_something, times); }
    double lucky_number() override { PYBIND11_OVERLOAD(double, B_Repeat, lucky_number, ); }
};
class PyC_Repeat : public C_Repeat {
public:
    using C_Repeat::C_Repeat;
    int unlucky_number() override { PYBIND11_OVERLOAD(int, C_Repeat, unlucky_number, ); }
    std::string say_something(unsigned times) override { PYBIND11_OVERLOAD(std::string, C_Repeat, say_something, times); }
    double lucky_number() override { PYBIND11_OVERLOAD(double, C_Repeat, lucky_number, ); }
};
class PyD_Repeat : public D_Repeat {
public:
    using D_Repeat::D_Repeat;
    int unlucky_number() override { PYBIND11_OVERLOAD(int, D_Repeat, unlucky_number, ); }
    std::string say_something(unsigned times) override { PYBIND11_OVERLOAD(std::string, D_Repeat, say_something, times); }
    double lucky_number() override { PYBIND11_OVERLOAD(double, D_Repeat, lucky_number, ); }
};

// Inheritance approach 2: templated trampoline classes.
//
// Advantages:
// - we have only 2 (template) class and 4 method declarations (one per virtual method, plus one for
//   any override of a pure virtual method), versus 4 classes and 6 methods (MI) or 4 classes and 11
//   methods (repeat).
// - Compared to MI, we also don't have to change the non-trampoline inheritance to virtual, and can
//   properly inherit constructors.
//
// Disadvantage:
// - the compiler must still generate and compile 14 different methods (more, even, than the 11
//   required for the repeat approach) instead of the 6 required for MI.  (If there was no pure
//   method (or no pure method override), the number would drop down to the same 11 as the repeat
//   approach).
template <class Base = A_Tpl>
class PyA_Tpl : public Base {
public:
    using Base::Base; // Inherit constructors
    int unlucky_number() override { PYBIND11_OVERLOAD_PURE(int, Base, unlucky_number, ); }
    std::string say_something(unsigned times) override { PYBIND11_OVERLOAD(std::string, Base, say_something, times); }
};
template <class Base = B_Tpl>
class PyB_Tpl : public PyA_Tpl<Base> {
public:
    using PyA_Tpl<Base>::PyA_Tpl; // Inherit constructors (via PyA_Tpl's inherited constructors)
    int unlucky_number() override { PYBIND11_OVERLOAD(int, Base, unlucky_number, ); }
    double lucky_number() override { PYBIND11_OVERLOAD(double, Base, lucky_number, ); }
};
// Since C_Tpl and D_Tpl don't declare any new virtual methods, we don't actually need these (we can
// use PyB_Tpl<C_Tpl> and PyB_Tpl<D_Tpl> for the trampoline classes instead):
/*
template <class Base = C_Tpl> class PyC_Tpl : public PyB_Tpl<Base> {
public:
    using PyB_Tpl<Base>::PyB_Tpl;
};
template <class Base = D_Tpl> class PyD_Tpl : public PyC_Tpl<Base> {
public:
    using PyC_Tpl<Base>::PyC_Tpl;
};
*/

void initialize_inherited_virtuals(py::module &m) {
    // test_inherited_virtuals

    // Method 1: repeat
    py::class_<A_Repeat, PyA_Repeat>(m, "A_Repeat")
        .def(py::init<>())
        .def("unlucky_number", &A_Repeat::unlucky_number)
        .def("say_something", &A_Repeat::say_something)
        .def("say_everything", &A_Repeat::say_everything);
    py::class_<B_Repeat, A_Repeat, PyB_Repeat>(m, "B_Repeat")
        .def(py::init<>())
        .def("lucky_number", &B_Repeat::lucky_number);
    py::class_<C_Repeat, B_Repeat, PyC_Repeat>(m, "C_Repeat")
        .def(py::init<>());
    py::class_<D_Repeat, C_Repeat, PyD_Repeat>(m, "D_Repeat")
        .def(py::init<>());

    // test_
    // Method 2: Templated trampolines
    py::class_<A_Tpl, PyA_Tpl<>>(m, "A_Tpl")
        .def(py::init<>())
        .def("unlucky_number", &A_Tpl::unlucky_number)
        .def("say_something", &A_Tpl::say_something)
        .def("say_everything", &A_Tpl::say_everything);
    py::class_<B_Tpl, A_Tpl, PyB_Tpl<>>(m, "B_Tpl")
        .def(py::init<>())
        .def("lucky_number", &B_Tpl::lucky_number);
    py::class_<C_Tpl, B_Tpl, PyB_Tpl<C_Tpl>>(m, "C_Tpl")
        .def(py::init<>());
    py::class_<D_Tpl, C_Tpl, PyB_Tpl<D_Tpl>>(m, "D_Tpl")
        .def(py::init<>());


    // Fix issue #1454 (crash when acquiring/releasing GIL on another thread in Python 2.7)
    m.def("test_gil", &test_gil);
    m.def("test_gil_from_thread", &test_gil_from_thread);
};
