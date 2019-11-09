#include <torch/custom_class.h>

#include <iostream>
#include <string>
#include <vector>

namespace torch {
namespace jit {

namespace {

struct Foo : torch::jit::CustomClassHolder {
  int x, y;
  Foo() : x(0), y(0) {}
  Foo(int x_, int y_) : x(x_), y(y_) {}
  int64_t info() {
    return this->x * this->y;
  }
  int64_t add(int64_t z) {
    return (x + y) * z;
  }
  void increment(int64_t z) {
    this->x += z;
    this->y += z;
  }
  int64_t combine(c10::intrusive_ptr<Foo> b) {
    return this->info() + b->info();
  }
  ~Foo() {
    // std::cout<<"Destroying object with values: "<<x<<' '<<y<<std::endl;
  }
};

template <class T>
struct Stack : torch::jit::CustomClassHolder {
  std::vector<T> stack_;
  Stack(std::vector<T> init) : stack_(init.begin(), init.end()) {}

  void push(T x) {
    stack_.push_back(x);
  }
  T pop() {
    auto val = stack_.back();
    stack_.pop_back();
    return val;
  }
};

static auto test = torch::jit::class_<Foo>("Foo")
                       .def(torch::jit::init<int64_t, int64_t>())
                       // .def(torch::jit::init<>())
                       .def("info", &Foo::info)
                       .def("increment", &Foo::increment)
                       .def("add", &Foo::add)
                       .def("combine", &Foo::combine);

static auto testStack = torch::jit::class_<Stack<std::string>>("StackString")
                            .def(torch::jit::init<std::vector<std::string>>())
                            .def("push", &Stack<std::string>::push)
                            .def("pop", &Stack<std::string>::pop);

} // namespace

} // namespace jit
} // namespace torch
