#include <ATen/core/GraphImplInterface.h>

namespace at {

C10_DEFINE_REGISTRY(GraphImplRegistry, GraphImplInterface, GraphImplArgs)

} // namespace at
