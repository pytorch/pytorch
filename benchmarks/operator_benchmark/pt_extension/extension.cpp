#include <torch/extension.h>
#include <torch/script.h>

using torch::Tensor;

Tensor consume(Tensor a) {
  return a;
}

// When JIT tracing is used on function with constant for loop,
// the for loop is optimized away because of dead code elimination.
// That caused an issue for our op benchmark which needs to run an op
// in a loop and report the execution time. This diff resolves that issue by
// registering this consume op with correct alias information which is DEFAULT.
auto reg = torch::RegisterOperators()
  .op("operator_benchmark::_consume", &consume);

PYBIND11_MODULE(cpp_extension, m) {
  m.def("_consume", &consume, "consume");
}
