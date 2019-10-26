#include "test/cpp/jit/test_base.h"
#include "test/cpp/jit/test_utils.h"

#include "torch/custom_class.h"

namespace torch {
namespace jit {

void testCustomClass() {
  class Foo : public CustomClassHolder {
   public:
    Foo(std::string name) : name(std::move(name)) {}

    std::string get_name() {
      return name;
    }

   private:
    std::string name;
  };

  torch::jit::class_<Foo>("Foo")
      .def(torch::jit::init<std::string>())
      .def("get_name", &Foo::get_name);
}

} // namespace jit
} // namespace torch
