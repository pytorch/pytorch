#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/native/hip/RocFFTCallbacks.h>

#include <ATen/native/hip/RocFFTPlanCache.h>
#include <c10/util/Exception.h>
#include <c10/util/env.h>

#ifdef USE_ROCFFT_CALLBACKS
#include <ATen/native/hip/rocfft_callbacks_spirv.h>
#endif

namespace at::native::detail {

#ifdef USE_ROCFFT_CALLBACKS

namespace {

const char* rocfft_callback_symbol(RocFFTCallbackKind kind) {
  switch (kind) {
    case RocFFTCallbackKind::LoadGather:
      return "at_rocfft_load_gather";
    case RocFFTCallbackKind::LoadGatherPow2:
      return "at_rocfft_load_gather_pow2";
    case RocFFTCallbackKind::StoreWindow:
      return "at_rocfft_store_window";
    case RocFFTCallbackKind::StoreOverlapAdd:
      return "at_rocfft_store_overlap_add";
    case RocFFTCallbackKind::None:
      break;
  }
  TORCH_INTERNAL_ASSERT(false, "no rocFFT callback symbol for kind ", static_cast<int>(kind));
}

bool is_load_callback(RocFFTCallbackKind kind) {
  return kind == RocFFTCallbackKind::LoadGather || kind == RocFFTCallbackKind::LoadGatherPow2;
}

} // namespace (anonymous)

void rocfft_register_callback(rocfft_plan_description desc, RocFFTCallbackKind kind) {
  // The two setters take the same arguments and differ only in which end of the
  // transform they hook.
  auto set_callback = is_load_callback(kind) ? rocfft_plan_description_set_load_callback
                                             : rocfft_plan_description_set_store_callback;
  ROCFFT_CHECK(set_callback(desc, rocfft_callback_symbol(kind), kRocFFTCallbackSPIRV,
      kRocFFTCallbackSPIRVSize, /*shared_mem_bytes=*/0));
}

void rocfft_set_callback_data(rocfft_execution_info info, RocFFTCallbackKind kind, void* data) {
  auto set_data = is_load_callback(kind) ? rocfft_execution_info_set_load_callback_data
                                         : rocfft_execution_info_set_store_callback_data;
  // One pointer per brick, and these plans are all single-brick.
  ROCFFT_CHECK(set_data(info, &data, 1));
}

bool rocfft_callbacks_available() {
  static const bool available = [] {
    if (c10::utils::check_env("TORCH_ROCM_DISABLE_ROCFFT_CALLBACKS") == true) {
      return false;
    }
    lazy_init_rocfft();
    // Plan creation is what compiles the callback, so a throwaway plan answers
    // the question without anything having to run on the device. Shaped like the
    // transform stft issues, to keep the probe on the path it is predicting.
    RocFFTParams params(/*signal_size=*/64, /*batch=*/1, /*in_stride=*/1, /*in_distance=*/64,
        /*out_stride=*/1, /*out_distance=*/33, RocFFTTransformType::R2C, /*forward=*/true,
        ScalarType::Float, RocFFTCallbackKind::LoadGather);
    try {
      RocFFTConfig probe(params);
    } catch (const c10::Error&) {
      return false;
    }
    return true;
  }();
  return available;
}

#else

void rocfft_register_callback(rocfft_plan_description, RocFFTCallbackKind) {
  TORCH_INTERNAL_ASSERT(false, "PyTorch was built without rocFFT callback support");
}

void rocfft_set_callback_data(rocfft_execution_info, RocFFTCallbackKind, void*) {
  TORCH_INTERNAL_ASSERT(false, "PyTorch was built without rocFFT callback support");
}

bool rocfft_callbacks_available() {
  return false;
}

#endif

} // namespace at::native::detail
