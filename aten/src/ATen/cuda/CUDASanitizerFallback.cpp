#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/impl/GPUTrace.h>
#include <torch/library.h>
#include <iostream>

void process_tensor(const at::Tensor& tensor, std::vector<uintptr_t>& vec) {
    c10::Storage storage = tensor.storage();
    if (storage.device_type() == c10::DeviceType::CUDA) {
      vec.push_back(reinterpret_cast<uintptr_t>(storage.data_ptr().get()));
    }
}

void process_ivalue(const c10::IValue& ivalue, std::vector<uintptr_t>& vec) {
    if (ivalue.isTensor()) {
        process_tensor(ivalue.toTensor(), vec);
    }
    else if (ivalue.isTensorList()) {
      for (const at::Tensor& tensor: ivalue.toTensorList()) {
        process_tensor(tensor, vec);
      }
    }
}

void CUDASanitizerFallback(const c10::OperatorHandle& op, c10::DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
    std::vector<uintptr_t> inputs, outputs;
    const c10::FunctionSchema& schema = op.schema();

    const std::size_t num_arguments = schema.arguments().size();
    for (const c10::IValue& ivalue: torch::jit::last(*stack, num_arguments)) {
      process_ivalue(ivalue, inputs);
    }

    op.redispatchBoxed(dispatch_keys & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::CUDASanitizer), stack);

    const std::size_t num_returns = schema.returns().size();
    for (const c10::IValue& ivalue: torch::jit::last(*stack, num_returns)) {
      process_ivalue(ivalue, outputs);
    }

    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    TORCH_INTERNAL_ASSERT(interp);
    (*interp)->trace_kernel_launch(schema, inputs, outputs);
}

TORCH_LIBRARY_IMPL(_, CUDASanitizer, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&CUDASanitizerFallback>());
}
