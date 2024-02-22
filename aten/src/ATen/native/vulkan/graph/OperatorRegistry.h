#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/graph/Graph.h>

#include <functional>
#include <unordered_map>

namespace at {
namespace native {
namespace vulkan {

using OpFunction = const std::function<at::native::vulkan::ValueRef(
    at::native::vulkan::ComputeGraph&,
    const std::vector<at::native::vulkan::ValueRef>&)>; // TODO: Generalize to
                                                        // support float,
                                                        // int64_t.

bool hasOpsFn(const std::string& name);

OpFunction& getOpsFn(const std::string& name);

// The Vulkan operator registry is a simplified version of
// fbcode/executorch/runtime/kernel/operator_registry.h
// that uses the C++ Standard Library.
class OperatorRegistry {
 public:
  static OperatorRegistry& getInstance();

  bool hasOpsFn(const std::string& name);
  OpFunction& getOpsFn(const std::string& name);

  OperatorRegistry(const OperatorRegistry&) = delete;
  OperatorRegistry(OperatorRegistry&&) = delete;
  OperatorRegistry& operator=(const OperatorRegistry&) = delete;
  OperatorRegistry& operator=(OperatorRegistry&&) = delete;

 private:
  // TODO: Input string corresponds to target_name. We may need to pass kwargs.
  using OpTable = std::unordered_map<std::string, OpFunction>;
  // @lint-ignore CLANGTIDY facebook-hte-NonPodStaticDeclaration
  static const OpTable kTable;

  OperatorRegistry() = default;
  ~OperatorRegistry() = default;
};

} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
