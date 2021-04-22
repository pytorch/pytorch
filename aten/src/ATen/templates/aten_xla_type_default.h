// ${generated_comment}

#include <ATen/Tensor.h>
#include <c10/core/Stream.h>

using c10::Stream;

namespace ${cpp_namespace} {

class AtenXlaTypeDefault {
 public:
${dispatch_aten_fallback_declarations}

};

// TODO: maybe kill this, doesn't look like XLA actually calls it anywhere
void RegisterAtenTypeFunctions();

}  // namespace torch_xla
