/*
    tests/test_custom-exceptions.cpp -- exception translation

    Copyright (c) 2016 Pim Schellart <P.Schellart@princeton.edu>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"

// A type that should be raised as an exception in Python
class MyException : public std::exception {
public:
    explicit MyException(const char * m) : message{m} {}
    virtual const char * what() const noexcept override {return message.c_str();}
private:
    std::string message = "";
};

// A type that should be translated to a standard Python exception
class MyException2 : public std::exception {
public:
    explicit MyException2(const char * m) : message{m} {}
    virtual const char * what() const noexcept override {return message.c_str();}
private:
    std::string message = "";
};

// A type that is not derived from std::exception (and is thus unknown)
class MyException3 {
public:
    explicit MyException3(const char * m) : message{m} {}
    virtual const char * what() const noexcept {return message.c_str();}
private:
    std::string message = "";
};

// A type that should be translated to MyException
// and delegated to its exception translator
class MyException4 : public std::exception {
public:
    explicit MyException4(const char * m) : message{m} {}
    virtual const char * what() const noexcept override {return message.c_str();}
private:
    std::string message = "";
};


// Like the above, but declared via the helper function
class MyException5 : public std::logic_error {
public:
    explicit MyException5(const std::string &what) : std::logic_error(what) {}
};

// Inherits from MyException5
class MyException5_1 : public MyException5 {
    using MyException5::MyException5;
};

struct PythonCallInDestructor {
    PythonCallInDestructor(const py::dict &d) : d(d) {}
    ~PythonCallInDestructor() { d["good"] = true; }

    py::dict d;
};

TEST_SUBMODULE(exceptions, m) {
    m.def("throw_std_exception", []() {
        throw std::runtime_error("This exception was intentionally thrown.");
    });

    // make a new custom exception and use it as a translation target
    static py::exception<MyException> ex(m, "MyException");
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const MyException &e) {
            // Set MyException as the active python error
            ex(e.what());
        }
    });

    // register new translator for MyException2
    // no need to store anything here because this type will
    // never by visible from Python
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const MyException2 &e) {
            // Translate this exception to a standard RuntimeError
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
    });

    // register new translator for MyException4
    // which will catch it and delegate to the previously registered
    // translator for MyException by throwing a new exception
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const MyException4 &e) {
            throw MyException(e.what());
        }
    });

    // A simple exception translation:
    auto ex5 = py::register_exception<MyException5>(m, "MyException5");
    // A slightly more complicated one that declares MyException5_1 as a subclass of MyException5
    py::register_exception<MyException5_1>(m, "MyException5_1", ex5.ptr());

    m.def("throws1", []() { throw MyException("this error should go to a custom type"); });
    m.def("throws2", []() { throw MyException2("this error should go to a standard Python exception"); });
    m.def("throws3", []() { throw MyException3("this error cannot be translated"); });
    m.def("throws4", []() { throw MyException4("this error is rethrown"); });
    m.def("throws5", []() { throw MyException5("this is a helper-defined translated exception"); });
    m.def("throws5_1", []() { throw MyException5_1("MyException5 subclass"); });
    m.def("throws_logic_error", []() { throw std::logic_error("this error should fall through to the standard handler"); });
    m.def("exception_matches", []() {
        py::dict foo;
        try { foo["bar"]; }
        catch (py::error_already_set& ex) {
            if (!ex.matches(PyExc_KeyError)) throw;
        }
    });

    m.def("throw_already_set", [](bool err) {
        if (err)
            PyErr_SetString(PyExc_ValueError, "foo");
        try {
            throw py::error_already_set();
        } catch (const std::runtime_error& e) {
            if ((err && e.what() != std::string("ValueError: foo")) ||
                (!err && e.what() != std::string("Unknown internal error occurred")))
            {
                PyErr_Clear();
                throw std::runtime_error("error message mismatch");
            }
        }
        PyErr_Clear();
        if (err)
            PyErr_SetString(PyExc_ValueError, "foo");
        throw py::error_already_set();
    });

    m.def("python_call_in_destructor", [](py::dict d) {
        try {
            PythonCallInDestructor set_dict_in_destructor(d);
            PyErr_SetString(PyExc_ValueError, "foo");
            throw py::error_already_set();
        } catch (const py::error_already_set&) {
            return true;
        }
        return false;
    });

    // test_nested_throws
    m.def("try_catch", [m](py::object exc_type, py::function f, py::args args) {
        try { f(*args); }
        catch (py::error_already_set &ex) {
            if (ex.matches(exc_type))
                py::print(ex.what());
            else
                throw;
        }
    });

}
