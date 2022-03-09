#include "lazy_tensor_core/csrc/ts_backend/aten_eager_fallback.h"

#include <torch/csrc/lazy/backend/backend_interface.h>
#include <torch/csrc/lazy/core/metrics.h>

#include <unordered_map>

#include "lazy_tensor_core/csrc/function_call_tracker.h"
#include "lazy_tensor_core/csrc/ts_backend/EagerFallback.h"

namespace torch_lazy_tensors {

static std::unordered_map<std::string, ::torch::lazy::Counter*>
    _eager_fallback_counters;

bool force_eager_fallback(c10::Symbol op) {
  static char* force_str = std::getenv("LTC_FORCE_FALLBACK");
  if (force_str != nullptr) {
    static auto force_sym = c10::Symbol::fromQualString(std::string(force_str));
    if (op == force_sym) {
      return true;
    }
  }
  return false;
}

void ltc_eager_fallback(const c10::OperatorHandle& op,
                        torch::jit::Stack* stack) {
  LTC_FN_TRACK(3);
  const auto name = c10::toString(op.operator_name());

  // Manually applying the TORCH_LAZY_COUNTER macro.
  // We need to do it ourselves and explicitly keep a mapping of counters
  // because this boxed fallback kernel is used by multiple operators,
  // and the macro stamps out a static Counter object with a fixed name
  // at the code location that it was called.
  if (_eager_fallback_counters.find(name) == _eager_fallback_counters.end()) {
    _eager_fallback_counters[name] = new ::torch::lazy::Counter(name);
  }
  _eager_fallback_counters[name]->AddValue(1);

  auto& args = op.schema().arguments();
  auto arguments = torch::jit::last(stack, args.size());

  // Log each tensor argument.
  for (int64_t idx = 0; idx < arguments.size(); ++idx) {
    const auto& ivalue = arguments[idx];
    if (ivalue.isTensor()) {
      VLOG(3) << ivalue.toTensor().toString();
    }
  }

  // Call the actual boxed CPU fallback.
  eager_fallback(op, stack,
                 torch::lazy::getBackend()->EagerFallbackDeviceType());
}

std::function<void(void)> register_ts_ltc_eager_fallback;

TORCH_LIBRARY_IMPL(_, Lazy, m) {
  register_ts_ltc_eager_fallback = [&]() {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&ltc_eager_fallback>());
  };
}

}  // namespace torch_lazy_tensors
