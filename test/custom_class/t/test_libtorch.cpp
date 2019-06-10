
#include <cassert>
#include <climits>
#include <cstring>
#include <iostream>
#include <iterator>
#include <list>
#include <torch/script.h>
#include <pybind11/pybind11.h>

using namespace std;

namespace py = pybind11;
void warp_perspective(torch::Tensor image) { cout<<"HEY"<<endl; }
struct Foo {
  int x, y;
  Foo(): x(2), y(5){
    std::cout<<"Running constructor"<<std::endl;
  }
  Foo(int x_, int y_) : x(x_), y(y_) {}
  void display() {
    cout<<"x: "<<x<<' '<<"y: "<<y<<endl;
  }
  int64_t add(int64_t z) {
    return (x+y)*z;
  }
  ~Foo() {
  }
};

static auto registry = torch::jit::RegisterOperators("my_ops::warp_perspective",
                                                     &warp_perspective);
static auto test = torch::jit::class_<Foo>("Foo")
                    // .init<>()
                    .init<int64_t, int64_t>()
                    .def("display", &Foo::display)
                    .def("add", &Foo::add);
