#include <torch/custom_class.h>
#include <torch/script.h>

namespace torch {
namespace jit {

struct ScalarTypeClass : public torch::CustomClassHolder {
  ScalarTypeClass(at::ScalarType s) : scalar_type_(s) {}
  at::ScalarType scalar_type_;
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
} // namespace jit
} // namespace torch
