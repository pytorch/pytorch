#ifndef CPU_UTILS_H
#define CPU_UTILS_H
#include <c10/macros/Export.h>

namespace at {
namespace native {

// Detect if CPU support Vector Neural Network Instruction.
TORCH_API bool is_cpu_support_vnni();

} // namespace native
} // namespace at

#endif // CPU_UTILS_H