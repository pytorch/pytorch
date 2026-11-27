#pragma once

#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/stableivalue_conversions.h>

#include <array>
#include <cstdint>
#include <type_traits>

namespace schema_adaptation_poc {

// All versions here are SIMULATED metadata; this fixture uses real 2.16
// headers.
constexpr uint64_t kPocV14 = 0x020e000000000000ULL;
constexpr uint64_t kPocV16 = 0x0210000000000000ULL;
constexpr uint64_t kPocV17 = 0x0211000000000000ULL;

struct RuntimeCapabilities {
  uint64_t version;
  bool has_dispatcher_v2;
  const char* op_name;
};

inline RuntimeCapabilities simulated_runtime(uint64_t version) {
  return {
      version,
      version >= kPocV16,
      version >= kPocV17 ? "_test_schema_poc_v17::example"
                         : "_test_schema_poc_v14::example"};
}

enum class ApiRoute { None, Legacy, DynamicV2 };

struct CallTrace {
  ApiRoute route = ApiRoute::None;
  size_t slots = 0;
  uint64_t build = 0;
  uint64_t target =
      0; // Unknown for the legacy API, not an inferred old target.
};

using DispatcherV2 = AOTITorchError (*)(
    const char*,
    const char*,
    StableIValue*,
    uint64_t,
    uint64_t);

// Models future TORCH_ABI_VERSION / TORCH_FEATURE_VERSION wrapper generation.
// It does not change the actual compiler's version macros or public ops.h.
template <uint64_t Build, uint64_t Target>
class Example {
 public:
  static_assert(Target != 0 && Target <= Build, "target must be <= build");

  Example(RuntimeCapabilities runtime, CallTrace& trace)
      : runtime_(runtime), trace_(trace) {}

  template <
      uint64_t T = Target,
      std::enable_if_t<T == Target && (T < kPocV17), int> = 0>
  torch::stable::Tensor example(const torch::stable::Tensor& x, int64_t n) {
    return invoke(x, n, false);
  }

  template <
      uint64_t T = Target,
      std::enable_if_t<T == Target && (T >= kPocV17), int> = 0>
  torch::stable::Tensor example(
      const torch::stable::Tensor& x,
      int64_t n,
      bool new_mode = false) {
    return invoke(x, n, new_mode);
  }

 private:
  torch::stable::Tensor invoke(
      const torch::stable::Tensor& x,
      int64_t n,
      bool new_mode) {
    STD_TORCH_CHECK(runtime_.version >= Target, "runtime is older than target");
    DispatcherV2 v2 = nullptr;
    if constexpr (Build >= kPocV16) {
      if (runtime_.version >= kPocV16 && runtime_.has_dispatcher_v2) {
        v2 = reinterpret_cast<DispatcherV2>(
            torch::stable::detail::lookup_stable_symbol(
                "torch_call_dispatcher_v2"));
        STD_TORCH_CHECK(v2 != nullptr, "fixture requires dispatcher v2 symbol");
      }
    }
    STD_TORCH_CHECK(
        !(Build >= kPocV17 && Target < kPocV17 &&
          runtime_.version >= kPocV17) ||
            v2 != nullptr,
        "target-gated source schema requires dispatcher v2");
    constexpr size_t slots = Target >= kPocV17 ? 3 : 2;
    trace_ = {
        v2 ? ApiRoute::DynamicV2 : ApiRoute::Legacy,
        slots,
        Build,
        v2 ? Target : 0};
    std::array<StableIValue, slots> stack{};
    stack[0] = torch::stable::detail::from(x); // New owning handle.
    stack[1] = torch::stable::detail::from(n);
    if constexpr (slots == 3) {
      stack[2] = torch::stable::detail::from(new_mode);
    }
    // Both C routes consume the boxed input handle. Only the return is unboxed.
    const auto err = v2
        ? v2(runtime_.op_name, "", stack.data(), Build, Target)
        : torch_call_dispatcher(runtime_.op_name, "", stack.data(), Build);
    STABLE_TORCH_ERROR_CODE_CHECK(err);
    return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
  }

  RuntimeCapabilities runtime_;
  CallTrace& trace_;
};

} // namespace schema_adaptation_poc
