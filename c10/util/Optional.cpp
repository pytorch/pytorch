#include <c10/util/Optional.h>

#include <type_traits>

// CUDA 9.2 and below fail while trying to compile default move constructor
// see https://github.com/pytorch/csprng/issues/84
#if (!defined(__CUDA_ARCH__) || !defined(CUDA_VERSION) || CUDA_VERSION > 9200)
static_assert(C10_IS_TRIVIALLY_COPYABLE(c10::optional<int>), "c10::optional<int> should be trivially copyable");
static_assert(C10_IS_TRIVIALLY_COPYABLE(c10::optional<bool>), "c10::optional<bool> should be trivially copyable");
#endif
