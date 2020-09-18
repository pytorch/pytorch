#include <string>

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void callgrind_block() {
    /*
    `eval(expr)` is a magical eval defined elsewhere. This method
    works because py::exec shares the same interpreter (and therefore globals)
    as the caller.
    */
    py::exec(R"(
        eval(expr)
    )");

}


PYBIND11_MODULE(bindings, m) {
    m.def("callgrind_block", &callgrind_block);
}
