#include <torch/custom_class.h>
#include <torch/script.h>

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

  c10::intrusive_ptr<Stack> clone() const {
    return c10::make_intrusive<Stack>(stack_);
  }

  void merge(const c10::intrusive_ptr<Stack>& c) {
    for (auto& elem : c->stack_) {
      push(elem);
    }
  }

  std::tuple<double, int64_t> return_a_tuple() const {
    return std::make_tuple(1337.0f, 123);
  }
};

struct PickleTester : torch::jit::CustomClassHolder {
  PickleTester(std::vector<int64_t> vals) : vals(std::move(vals)) {}
  std::vector<int64_t> vals;
};

static auto test = torch::jit::class_<Foo>("_TorchScriptTesting_Foo")
                       .def(torch::jit::init<int64_t, int64_t>())
                       // .def(torch::jit::init<>())
                       .def("info", &Foo::info)
                       .def("increment", &Foo::increment)
                       .def("add", &Foo::add)
                       .def("combine", &Foo::combine);

static auto testStack =
    torch::jit::class_<Stack<std::string>>("_TorchScriptTesting_StackString")
        .def(torch::jit::init<std::vector<std::string>>())
        .def("push", &Stack<std::string>::push)
        .def("pop", &Stack<std::string>::pop)
        .def("clone", &Stack<std::string>::clone)
        .def("merge", &Stack<std::string>::merge)
        .def_pickle(
            [](const c10::intrusive_ptr<Stack<std::string>>& self) {
              return self->stack_;
            },
            [](std::vector<std::string> state) { // __setstate__
              return c10::make_intrusive<Stack<std::string>>(
                  std::vector<std::string>{"i", "was", "deserialized"});
            })
        .def("return_a_tuple", &Stack<std::string>::return_a_tuple)
        .def(
            "top",
            [](const c10::intrusive_ptr<Stack<std::string>>& self)
                -> std::string { return self->stack_.back(); });
// clang-format off
        // The following will fail with a static assert telling you you have to
        // take an intrusive_ptr<Stack> as the first argument.
        // .def("foo", [](int64_t a) -> int64_t{ return 3;});
// clang-format on

static auto testPickle =
    torch::jit::class_<PickleTester>("_TorchScriptTesting_PickleTester")
        .def(torch::jit::init<std::vector<int64_t>>())
        .def_pickle(
            [](c10::intrusive_ptr<PickleTester> self) { // __getstate__
              return std::vector<int64_t>{1, 3, 3, 7};
            },
            [](std::vector<int64_t> state) { // __setstate__
              return c10::make_intrusive<PickleTester>(std::move(state));
            })
        .def(
            "top",
            [](const c10::intrusive_ptr<PickleTester>& self) {
              return self->vals.back();
            })
        .def("pop", [](const c10::intrusive_ptr<PickleTester>& self) {
          auto val = self->vals.back();
          self->vals.pop_back();
          return val;
        });

at::Tensor take_an_instance(const c10::intrusive_ptr<PickleTester>& instance) {
  return torch::zeros({instance->vals.back(), 4});
}

torch::RegisterOperators& register_take_instance() {
  static auto instance_registry = torch::RegisterOperators().op(
  torch::RegisterOperators::options()
      .schema(
          "_TorchScriptTesting::take_an_instance(__torch__.torch.classes._TorchScriptTesting_PickleTester x) -> Tensor Y")
      .catchAllKernel<decltype(take_an_instance), &take_an_instance>());
  return instance_registry;
}

static auto& ensure_take_instance_registered = register_take_instance();


} // namespace

} // namespace jit
} // namespace torch
