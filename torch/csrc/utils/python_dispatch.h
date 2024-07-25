#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::impl::dispatch {

void initDispatchBindings(PyObject* module);

void python_op_registration_trampoline_impl(
    const c10::OperatorHandle& op,
    c10::DispatchKey key,
    c10::DispatchKeySet keyset,
    torch::jit::Stack* stack,
    bool with_keyset);

} // namespace torch::impl::dispatch
