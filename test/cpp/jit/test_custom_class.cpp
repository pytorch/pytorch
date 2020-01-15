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
  // Note: there are some issues with returning tuples from custom ops currently,
  // thus using string
  std::string __getstate__() {
    std::stringstream ss;
    ss << "magickey " << x << " " << y;
    return ss.str();
  }
  void __setstate__(const std::string& data) {
    std::stringstream ss(data);
    std::string magic;
    ss >> magic >> x >> y;
    TORCH_CHECK(magic == "magickey");
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

static auto test = torch::jit::class_<Foo>("_TorchScriptTesting_Foo")
                       .def(torch::jit::init<int64_t, int64_t>())
                       // TODO: multiple overloads of the method are not supported because of different override names
                       // .def(torch::jit::init<>())
                       .def("info", &Foo::info)
                       .def("increment", &Foo::increment)
                       .def("add", &Foo::add)
                       .def("combine", &Foo::combine)
                       // TODO: we might need a different mechanism for binding __setstate__ such that it also constructs
                       // the object. See how pybind11 does it:
                       // https://pybind11.readthedocs.io/en/stable/advanced/classes.html#pickling-support
                       .def("__getstate__", &Foo::__getstate__)
                       .def("__setstate__", &Foo::__setstate__);

static auto testStack =
    torch::jit::class_<Stack<std::string>>("_TorchScriptTesting_StackString")
        .def(torch::jit::init<std::vector<std::string>>())
        .def("push", &Stack<std::string>::push)
        .def("pop", &Stack<std::string>::pop);

torch::RegisterOperators reg(
    "_TorchScriptTesting::standalone_multiply_mutable", [](int64_t factor, c10::intrusive_ptr<Foo> arg) {
      arg->x *= factor;
      arg->y *= factor;
    });

} // namespace

} // namespace jit
} // namespace torch
