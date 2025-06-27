#include <c10/util/Exception.h>
#include <nvshmem.h>

namespace c10d::nvshmem_extension {

#define NVSHMEM_CHECK(stmt, msg)                      \
  do {                                                \
    int result = (stmt);                              \
    if (NVSHMEMX_SUCCESS != result) {                 \
      std::string err = std::string(__FILE__) + ":" + \
          std::to_string(__LINE__) + " " + msg +      \
          ". Error code: " + std::to_string(result);  \
      TORCH_CHECK(false, err);                        \
    }                                                 \
  } while (0)

} // namespace c10d::nvshmem_extension
