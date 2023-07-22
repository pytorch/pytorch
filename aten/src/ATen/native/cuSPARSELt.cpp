#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/CUDASparse.h>
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/CallOnce.h>
#include <c10/util/Half.h>
#include <ATen/cuda/CUDAConfig.h>
#include <cusparse.h>
#include <cstdint>


namespace at {
namespace native {

// #if defined(USE_CUSPARSELT)
// cusparseLtHandle_t handle;
// #endif

// static bool _init_cusparselt(){
//     static c10::once_flag once_;
//     static bool cusparselt_successfully_initialized_ = false;

//     c10::call_once(once_, []() {
//         // Is there a better way to do this?
//         TORCH_CUDASPARSE_CHECK(cusparseLtInit(&handle));
//         cusparselt_successfully_initialized_ = true;
//     });
//     return cusparselt_successfully_initialized_;
// }

bool _cslt_available(){
    return AT_CUSPARSELT_ENABLED();
}

} // namespace native
} // namespace at
