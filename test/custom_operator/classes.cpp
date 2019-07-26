
#include <cassert>
#include <climits>
#include <cstring>
#include <iostream>
#include <iterator>
#include <list>
#include <torch/script.h>
#include <torch/custom_class.h>
#include <pybind11/pybind11.h>

using namespace std;

namespace py = pybind11;

struct Foo : torch::jit::torchbind_class {
  int x, y;
  Foo(): x(2), y(5){}
  Foo(int x_, int y_) : x(x_), y(y_) {}
  int64_t info() {
    return this->x * this->y;
  }
  int64_t add(int64_t z) {
    return (x+y)*z;
  }
  void increment(int64_t z) {
    this->x+=z;
    this->y+=z;
  }
  int64_t combine(c10::intrusive_ptr<Foo> b) {
    return this->info() + b->info();
  }
  ~Foo() {
    // std::cout<<"Destroying object with values: "<<x<<' '<<y<<std::endl;
  }
};

static auto test = torch::jit::class_<Foo>("Foo")
                    .def(torch::jit::init<int64_t, int64_t>())
                    // .def(torch::jit::init<>())
                    .def("info", &Foo::info)
                    .def("increment", &Foo::increment)
                    // .def("add", &Foo::add);
                    .def("combine", &Foo::combine)
                    ;
