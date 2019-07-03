#include <c10/util/intrusive_ptr.h>

namespace c10 {

std::atomic<int64_t> intrusive_ptr_target::unique_id_counter_{0};

}  // namespace c10
