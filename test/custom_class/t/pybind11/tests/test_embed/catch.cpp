// The Catch implementation is compiled here. This is a standalone
// translation unit to avoid recompiling it for every test change.

#include <pybind11/embed.h>

#ifdef _MSC_VER
// Silence MSVC C++17 deprecation warning from Catch regarding std::uncaught_exceptions (up to catch
// 2.0.1; this should be fixed in the next catch release after 2.0.1).
#  pragma warning(disable: 4996)
#endif

#define CATCH_CONFIG_RUNNER
#include <catch.hpp>

namespace py = pybind11;

int main(int argc, char *argv[]) {
    py::scoped_interpreter guard{};
    auto result = Catch::Session().run(argc, argv);

    return result < 0xff ? result : 0xff;
}
