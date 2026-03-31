// Windows ROCm ABI Bridge - CPU side (MSVC compiled)
// Extracts raw pointers from ArrayRef/std::optional and calls bridge functions in torch_hip.dll

#if defined(_WIN32) && defined(USE_ROCM)

#include <ATen/core/Tensor.h>
#include <c10/core/ScalarType.h>
#include <c10/util/ArrayRef.h>
#include <ATen/native/cuda/WindowsROCmBridge.h>

namespace at::native {

// Bridge functions in torch_hip.dll (Clang compiled)
extern "C" {
Tensor& _scaled_mm_cuda_v2_bridge(
    const Tensor& a, const Tensor& b,
    const Tensor* sa, int64_t sa_n, const int64_t* ra, int64_t ra_n, const int64_t* wa, int64_t wa_n,
    const Tensor* sb, int64_t sb_n, const int64_t* rb, int64_t rb_n, const int64_t* wb, int64_t wb_n,
    const Tensor* bias, const c10::ScalarType* dtype, const int64_t* cd, int64_t cd_n,
    bool fast_accum, Tensor& out);

Tensor _scaled_mm_cuda_v2_functional_bridge(
    const Tensor& a, const Tensor& b,
    const Tensor* sa, int64_t sa_n, const int64_t* ra, int64_t ra_n, const int64_t* wa, int64_t wa_n,
    const Tensor* sb, int64_t sb_n, const int64_t* rb, int64_t rb_n, const int64_t* wb, int64_t wb_n,
    const Tensor* bias, const c10::ScalarType* dtype, const int64_t* cd, int64_t cd_n,
    bool fast_accum);

Tensor _scaled_grouped_mm_cuda_v2_bridge(
    const Tensor& a, const Tensor& b,
    const Tensor* sa, int64_t sa_n, const int64_t* ra, int64_t ra_n, const int64_t* wa, int64_t wa_n,
    const Tensor* sb, int64_t sb_n, const int64_t* rb, int64_t rb_n, const int64_t* wb, int64_t wb_n,
    const Tensor* offs, const Tensor* bias, const c10::ScalarType* dtype,
    const int64_t* cd, int64_t cd_n, bool fast_accum);

Tensor _grouped_mm_cuda_bridge(
    const Tensor& a, const Tensor& b,
    const Tensor* offs, const Tensor* bias, const c10::ScalarType* dtype);
}

// CPU-side wrappers - extract raw pointers and call bridges
static Tensor& _scaled_mm_cuda_v2_out_wrapper(
    const Tensor& a, const Tensor& b,
    ArrayRef<Tensor> sa, IntArrayRef ra, IntArrayRef wa,
    ArrayRef<Tensor> sb, IntArrayRef rb, IntArrayRef wb,
    const std::optional<Tensor>& bias, const std::optional<c10::ScalarType> dtype,
    IntArrayRef cd, bool fast_accum, Tensor& out) {
    return _scaled_mm_cuda_v2_bridge(a, b,
        EXTRACT_ARRAYREF(sa), EXTRACT_ARRAYREF(ra), EXTRACT_ARRAYREF(wa),
        EXTRACT_ARRAYREF(sb), EXTRACT_ARRAYREF(rb), EXTRACT_ARRAYREF(wb),
        EXTRACT_OPTIONAL(bias), EXTRACT_OPTIONAL(dtype), EXTRACT_ARRAYREF(cd),
        fast_accum, out);
}

static Tensor _scaled_mm_cuda_v2_wrapper(
    const Tensor& a, const Tensor& b,
    ArrayRef<Tensor> sa, IntArrayRef ra, IntArrayRef wa,
    ArrayRef<Tensor> sb, IntArrayRef rb, IntArrayRef wb,
    const std::optional<Tensor>& bias, const std::optional<c10::ScalarType> dtype,
    IntArrayRef cd, bool fast_accum) {
    return _scaled_mm_cuda_v2_functional_bridge(a, b,
        EXTRACT_ARRAYREF(sa), EXTRACT_ARRAYREF(ra), EXTRACT_ARRAYREF(wa),
        EXTRACT_ARRAYREF(sb), EXTRACT_ARRAYREF(rb), EXTRACT_ARRAYREF(wb),
        EXTRACT_OPTIONAL(bias), EXTRACT_OPTIONAL(dtype), EXTRACT_ARRAYREF(cd),
        fast_accum);
}

static Tensor _scaled_grouped_mm_cuda_v2_wrapper(
    const Tensor& a, const Tensor& b,
    ArrayRef<Tensor> sa, IntArrayRef ra, IntArrayRef wa,
    ArrayRef<Tensor> sb, IntArrayRef rb, IntArrayRef wb,
    const std::optional<Tensor>& offs, const std::optional<Tensor>& bias,
    const std::optional<c10::ScalarType> dtype, IntArrayRef cd, bool fast_accum) {
    return _scaled_grouped_mm_cuda_v2_bridge(a, b,
        EXTRACT_ARRAYREF(sa), EXTRACT_ARRAYREF(ra), EXTRACT_ARRAYREF(wa),
        EXTRACT_ARRAYREF(sb), EXTRACT_ARRAYREF(rb), EXTRACT_ARRAYREF(wb),
        EXTRACT_OPTIONAL(offs), EXTRACT_OPTIONAL(bias), EXTRACT_OPTIONAL(dtype),
        EXTRACT_ARRAYREF(cd), fast_accum);
}

static Tensor _grouped_mm_cuda_wrapper(
    const Tensor& a, const Tensor& b,
    const std::optional<Tensor>& offs, const std::optional<Tensor>& bias,
    std::optional<c10::ScalarType> dtype) {
    return _grouped_mm_cuda_bridge(a, b,
        EXTRACT_OPTIONAL(offs), EXTRACT_OPTIONAL(bias), EXTRACT_OPTIONAL(dtype));
}

}  // namespace at::native

#endif  // defined(_WIN32) && defined(USE_ROCM)
