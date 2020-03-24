#include <torch/custom_class.h>
#include <torch/script.h>

#include <iostream>
#include <string>
#include <vector>

namespace torch {
namespace jit {

namespace {

struct Foo : torch::CustomClassHolder {
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
struct MyStackClass : torch::CustomClassHolder {
  std::vector<T> stack_;
  MyStackClass(std::vector<T> init) : stack_(init.begin(), init.end()) {}

  void push(T x) {
    stack_.push_back(x);
  }
  T pop() {
    auto val = stack_.back();
    stack_.pop_back();
    return val;
  }

  c10::intrusive_ptr<MyStackClass> clone() const {
    return c10::make_intrusive<MyStackClass>(stack_);
  }

  void merge(const c10::intrusive_ptr<MyStackClass>& c) {
    for (auto& elem : c->stack_) {
      push(elem);
    }
  }

  std::tuple<double, int64_t> return_a_tuple() const {
    return std::make_tuple(1337.0f, 123);
  }
};

struct PickleTester : torch::CustomClassHolder {
  PickleTester(std::vector<int64_t> vals) : vals(std::move(vals)) {}
  std::vector<int64_t> vals;
};

static auto test = torch::class_<Foo>("_TorchScriptTesting_Foo")
                       .def(torch::init<int64_t, int64_t>())
                       // .def(torch::init<>())
                       .def("info", &Foo::info)
                       .def("increment", &Foo::increment)
                       .def("add", &Foo::add)
                       .def("combine", &Foo::combine);

static auto testStack =
    torch::class_<MyStackClass<std::string>>("_TorchScriptTesting_StackString")
        .def(torch::init<std::vector<std::string>>())
        .def("push", &MyStackClass<std::string>::push)
        .def("pop", &MyStackClass<std::string>::pop)
        .def("clone", &MyStackClass<std::string>::clone)
        .def("merge", &MyStackClass<std::string>::merge)
        .def_pickle(
            [](const c10::intrusive_ptr<MyStackClass<std::string>>& self) {
              return self->stack_;
            },
            [](std::vector<std::string> state) { // __setstate__
              return c10::make_intrusive<MyStackClass<std::string>>(
                  std::vector<std::string>{"i", "was", "deserialized"});
            })
        .def("return_a_tuple", &MyStackClass<std::string>::return_a_tuple)
        .def(
            "top",
            [](const c10::intrusive_ptr<MyStackClass<std::string>>& self)
                -> std::string { return self->stack_.back(); });
// clang-format off
        // The following will fail with a static assert telling you you have to
        // take an intrusive_ptr<MyStackClass> as the first argument.
        // .def("foo", [](int64_t a) -> int64_t{ return 3;});
// clang-format on

static auto testPickle =
    torch::class_<PickleTester>("_TorchScriptTesting_PickleTester")
        .def(torch::init<std::vector<int64_t>>())
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

void testTorchbindIValueAPI() {
  script::Module m("m");

  // test make_custom_class API
  auto custom_class_obj = make_custom_class<MyStackClass<std::string>>(
      std::vector<std::string>{"foo", "bar"});
  m.define(R"(
    def forward(self, s : __torch__.torch.classes._TorchScriptTesting_StackString):
      return s.pop(), s
  )");

  auto test_with_obj = [&m](IValue obj, std::string expected) {
    auto res = m.run_method("forward", obj);
    auto tup = res.toTuple();
    AT_ASSERT(tup->elements().size() == 2);
    auto str = tup->elements()[0].toStringRef();
    auto other_obj =
        tup->elements()[1].toCustomClass<MyStackClass<std::string>>();
    AT_ASSERT(str == expected);
    auto ref_obj = obj.toCustomClass<MyStackClass<std::string>>();
    AT_ASSERT(other_obj.get() == ref_obj.get());
  };

  test_with_obj(custom_class_obj, "bar");

  // test IValue() API
  auto my_new_stack = c10::make_intrusive<MyStackClass<std::string>>(
      std::vector<std::string>{"baz", "boo"});
  auto new_stack_ivalue = c10::IValue(my_new_stack);

  test_with_obj(new_stack_ivalue, "boo");
}

} // namespace jit
} // namespace torch
