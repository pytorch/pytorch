#include <pybind11/embed.h>

#ifdef _MSC_VER
// Silence MSVC C++17 deprecation warning from Catch regarding std::uncaught_exceptions (up to catch
// 2.0.1; this should be fixed in the next catch release after 2.0.1).
#  pragma warning(disable: 4996)
#endif

#include <catch.hpp>

#include <thread>
#include <fstream>
#include <functional>

namespace py = pybind11;
using namespace py::literals;

class Widget {
public:
    Widget(std::string message) : message(message) { }
    virtual ~Widget() = default;

    std::string the_message() const { return message; }
    virtual int the_answer() const = 0;

private:
    std::string message;
};

class PyWidget final : public Widget {
    using Widget::Widget;

    int the_answer() const override { PYBIND11_OVERLOAD_PURE(int, Widget, the_answer); }
};

PYBIND11_EMBEDDED_MODULE(widget_module, m) {
    py::class_<Widget, PyWidget>(m, "Widget")
        .def(py::init<std::string>())
        .def_property_readonly("the_message", &Widget::the_message);

    m.def("add", [](int i, int j) { return i + j; });
}

PYBIND11_EMBEDDED_MODULE(throw_exception, ) {
    throw std::runtime_error("C++ Error");
}

PYBIND11_EMBEDDED_MODULE(throw_error_already_set, ) {
    auto d = py::dict();
    d["missing"].cast<py::object>();
}

TEST_CASE("Pass classes and data between modules defined in C++ and Python") {
    auto module = py::module::import("test_interpreter");
    REQUIRE(py::hasattr(module, "DerivedWidget"));

    auto locals = py::dict("hello"_a="Hello, World!", "x"_a=5, **module.attr("__dict__"));
    py::exec(R"(
        widget = DerivedWidget("{} - {}".format(hello, x))
        message = widget.the_message
    )", py::globals(), locals);
    REQUIRE(locals["message"].cast<std::string>() == "Hello, World! - 5");

    auto py_widget = module.attr("DerivedWidget")("The question");
    auto message = py_widget.attr("the_message");
    REQUIRE(message.cast<std::string>() == "The question");

    const auto &cpp_widget = py_widget.cast<const Widget &>();
    REQUIRE(cpp_widget.the_answer() == 42);
}

TEST_CASE("Import error handling") {
    REQUIRE_NOTHROW(py::module::import("widget_module"));
    REQUIRE_THROWS_WITH(py::module::import("throw_exception"),
                        "ImportError: C++ Error");
    REQUIRE_THROWS_WITH(py::module::import("throw_error_already_set"),
                        Catch::Contains("ImportError: KeyError"));
}

TEST_CASE("There can be only one interpreter") {
    static_assert(std::is_move_constructible<py::scoped_interpreter>::value, "");
    static_assert(!std::is_move_assignable<py::scoped_interpreter>::value, "");
    static_assert(!std::is_copy_constructible<py::scoped_interpreter>::value, "");
    static_assert(!std::is_copy_assignable<py::scoped_interpreter>::value, "");

    REQUIRE_THROWS_WITH(py::initialize_interpreter(), "The interpreter is already running");
    REQUIRE_THROWS_WITH(py::scoped_interpreter(), "The interpreter is already running");

    py::finalize_interpreter();
    REQUIRE_NOTHROW(py::scoped_interpreter());
    {
        auto pyi1 = py::scoped_interpreter();
        auto pyi2 = std::move(pyi1);
    }
    py::initialize_interpreter();
}

bool has_pybind11_internals_builtin() {
    auto builtins = py::handle(PyEval_GetBuiltins());
    return builtins.contains(PYBIND11_INTERNALS_ID);
};

bool has_pybind11_internals_static() {
    auto **&ipp = py::detail::get_internals_pp();
    return ipp && *ipp;
}

TEST_CASE("Restart the interpreter") {
    // Verify pre-restart state.
    REQUIRE(py::module::import("widget_module").attr("add")(1, 2).cast<int>() == 3);
    REQUIRE(has_pybind11_internals_builtin());
    REQUIRE(has_pybind11_internals_static());
    REQUIRE(py::module::import("external_module").attr("A")(123).attr("value").cast<int>() == 123);

    // local and foreign module internals should point to the same internals:
    REQUIRE(reinterpret_cast<uintptr_t>(*py::detail::get_internals_pp()) ==
            py::module::import("external_module").attr("internals_at")().cast<uintptr_t>());

    // Restart the interpreter.
    py::finalize_interpreter();
    REQUIRE(Py_IsInitialized() == 0);

    py::initialize_interpreter();
    REQUIRE(Py_IsInitialized() == 1);

    // Internals are deleted after a restart.
    REQUIRE_FALSE(has_pybind11_internals_builtin());
    REQUIRE_FALSE(has_pybind11_internals_static());
    pybind11::detail::get_internals();
    REQUIRE(has_pybind11_internals_builtin());
    REQUIRE(has_pybind11_internals_static());
    REQUIRE(reinterpret_cast<uintptr_t>(*py::detail::get_internals_pp()) ==
            py::module::import("external_module").attr("internals_at")().cast<uintptr_t>());

    // Make sure that an interpreter with no get_internals() created until finalize still gets the
    // internals destroyed
    py::finalize_interpreter();
    py::initialize_interpreter();
    bool ran = false;
    py::module::import("__main__").attr("internals_destroy_test") =
        py::capsule(&ran, [](void *ran) { py::detail::get_internals(); *static_cast<bool *>(ran) = true; });
    REQUIRE_FALSE(has_pybind11_internals_builtin());
    REQUIRE_FALSE(has_pybind11_internals_static());
    REQUIRE_FALSE(ran);
    py::finalize_interpreter();
    REQUIRE(ran);
    py::initialize_interpreter();
    REQUIRE_FALSE(has_pybind11_internals_builtin());
    REQUIRE_FALSE(has_pybind11_internals_static());

    // C++ modules can be reloaded.
    auto cpp_module = py::module::import("widget_module");
    REQUIRE(cpp_module.attr("add")(1, 2).cast<int>() == 3);

    // C++ type information is reloaded and can be used in python modules.
    auto py_module = py::module::import("test_interpreter");
    auto py_widget = py_module.attr("DerivedWidget")("Hello after restart");
    REQUIRE(py_widget.attr("the_message").cast<std::string>() == "Hello after restart");
}

TEST_CASE("Subinterpreter") {
    // Add tags to the modules in the main interpreter and test the basics.
    py::module::import("__main__").attr("main_tag") = "main interpreter";
    {
        auto m = py::module::import("widget_module");
        m.attr("extension_module_tag") = "added to module in main interpreter";

        REQUIRE(m.attr("add")(1, 2).cast<int>() == 3);
    }
    REQUIRE(has_pybind11_internals_builtin());
    REQUIRE(has_pybind11_internals_static());

    /// Create and switch to a subinterpreter.
    auto main_tstate = PyThreadState_Get();
    auto sub_tstate = Py_NewInterpreter();

    // Subinterpreters get their own copy of builtins. detail::get_internals() still
    // works by returning from the static variable, i.e. all interpreters share a single
    // global pybind11::internals;
    REQUIRE_FALSE(has_pybind11_internals_builtin());
    REQUIRE(has_pybind11_internals_static());

    // Modules tags should be gone.
    REQUIRE_FALSE(py::hasattr(py::module::import("__main__"), "tag"));
    {
        auto m = py::module::import("widget_module");
        REQUIRE_FALSE(py::hasattr(m, "extension_module_tag"));

        // Function bindings should still work.
        REQUIRE(m.attr("add")(1, 2).cast<int>() == 3);
    }

    // Restore main interpreter.
    Py_EndInterpreter(sub_tstate);
    PyThreadState_Swap(main_tstate);

    REQUIRE(py::hasattr(py::module::import("__main__"), "main_tag"));
    REQUIRE(py::hasattr(py::module::import("widget_module"), "extension_module_tag"));
}

TEST_CASE("Execution frame") {
    // When the interpreter is embedded, there is no execution frame, but `py::exec`
    // should still function by using reasonable globals: `__main__.__dict__`.
    py::exec("var = dict(number=42)");
    REQUIRE(py::globals()["var"]["number"].cast<int>() == 42);
}

TEST_CASE("Threads") {
    // Restart interpreter to ensure threads are not initialized
    py::finalize_interpreter();
    py::initialize_interpreter();
    REQUIRE_FALSE(has_pybind11_internals_static());

    constexpr auto num_threads = 10;
    auto locals = py::dict("count"_a=0);

    {
        py::gil_scoped_release gil_release{};
        REQUIRE(has_pybind11_internals_static());

        auto threads = std::vector<std::thread>();
        for (auto i = 0; i < num_threads; ++i) {
            threads.emplace_back([&]() {
                py::gil_scoped_acquire gil{};
                locals["count"] = locals["count"].cast<int>() + 1;
            });
        }

        for (auto &thread : threads) {
            thread.join();
        }
    }

    REQUIRE(locals["count"].cast<int>() == num_threads);
}

// Scope exit utility https://stackoverflow.com/a/36644501/7255855
struct scope_exit {
    std::function<void()> f_;
    explicit scope_exit(std::function<void()> f) noexcept : f_(std::move(f)) {}
    ~scope_exit() { if (f_) f_(); }
};

TEST_CASE("Reload module from file") {
    // Disable generation of cached bytecode (.pyc files) for this test, otherwise
    // Python might pick up an old version from the cache instead of the new versions
    // of the .py files generated below
    auto sys = py::module::import("sys");
    bool dont_write_bytecode = sys.attr("dont_write_bytecode").cast<bool>();
    sys.attr("dont_write_bytecode") = true;
    // Reset the value at scope exit
    scope_exit reset_dont_write_bytecode([&]() {
        sys.attr("dont_write_bytecode") = dont_write_bytecode;
    });

    std::string module_name = "test_module_reload";
    std::string module_file = module_name + ".py";

    // Create the module .py file
    std::ofstream test_module(module_file);
    test_module << "def test():\n";
    test_module << "    return 1\n";
    test_module.close();
    // Delete the file at scope exit
    scope_exit delete_module_file([&]() {
        std::remove(module_file.c_str());
    });

    // Import the module from file
    auto module = py::module::import(module_name.c_str());
    int result = module.attr("test")().cast<int>();
    REQUIRE(result == 1);

    // Update the module .py file with a small change
    test_module.open(module_file);
    test_module << "def test():\n";
    test_module << "    return 2\n";
    test_module.close();

    // Reload the module
    module.reload();
    result = module.attr("test")().cast<int>();
    REQUIRE(result == 2);
}
