#include <torch/csrc/profiler/standalone/privateuse1_observer.h>

namespace torch {
namespace profiler {
namespace impl {

DEFINE_DISPATCH(pushPRIVATEUSE1CallbacksStub);
REGISTER_NO_CPU_DISPATCH(pushPRIVATEUSE1CallbacksStub);

} // namespace impl
} // namespace profiler
} // namespace torch