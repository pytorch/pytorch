/// Flush-To-Zero and Denormals-Are-Zero mode
///
/// Flush-To-Zero (FTZ) and Denormals-Are-Zero (DAZ) are modes that bypass
/// IEEE 754 methods of dealing with denormal floating-point numbers on x86-64
/// and some x86 CPUs. They result in reduced precision for values near zero,
/// but increased performance.
///
/// See https://software.intel.com/en-us/articles/x87-and-sse-floating-point-assists-in-ia-32-flush-to-zero-ftz-and-denormals-are-zero-daz

namespace at { namespace cpu {

bool set_flush_denormal(bool on);

}}  // namespace at::cpu
