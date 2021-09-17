#include "lazy_tensor_core/csrc/ts_backend/aten_eager_fallback.h"

#include <unordered_map>

#include "lazy_tensor_core/csrc/function_call_tracker.h"
#include "lazy_tensor_core/csrc/ts_backend/EagerFallback.h"
#include "lazy_tensor_core/csrc/ts_backend/ts_computation_client.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/ltc_logging.h"
#include "lazy_tensors/computation_client/metrics.h"

namespace torch_lazy_tensors {

static std::unordered_map<std::string, ::lazy_tensors::metrics::Counter*>
    _eager_fallback_counters;

void ltc_eager_fallback(const c10::OperatorHandle& op,
                        torch::jit::Stack* stack) {
  LTC_FN_TRACK(3);
  const auto name = c10::toString(op.operator_name());

  // Manually applying the LTC_COUNTER macro.
  // We need to do it ourselves and explicitly keep a mapping of counters
  // because this boxed fallback kernel is used by multiple operators,
  // and the macro stamps out a static Counter object with a fixed name
  // at the code location that it was called.
  if (_eager_fallback_counters.find(name) == _eager_fallback_counters.end()) {
    _eager_fallback_counters[name] = new ::lazy_tensors::metrics::Counter(name);
  }
  _eager_fallback_counters[name]->AddValue(1);

  auto& args = op.schema().arguments();
  auto arguments = torch::jit::last(stack, args.size());

  // Log each tensor argument.
  for (int64_t idx = 0; idx < arguments.size(); ++idx) {
    const auto& ivalue = arguments[idx];
    if (ivalue.isTensor()) {
      LTC_VLOG(3) << ivalue.toTensor().toString();
    }
  }

  // Call the actual boxed CPU fallback.
  eager_fallback(
      op, stack,
      lazy_tensors::compiler::TSComputationClient::HardwareDeviceType());
}

TORCH_LIBRARY_IMPL(_, Lazy, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&ltc_eager_fallback>());
}

}  // namespace torch_lazy_tensors
