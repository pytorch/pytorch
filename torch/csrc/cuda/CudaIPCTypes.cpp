#include <torch/csrc/cuda/CudaIPCTypes.h>

namespace c10 {
namespace {
REGISTER_FREE_MEMORY_CALLBACK("cuda_ipc_collect", CudaIPCCollectCallback)
}
} // namespace c10
