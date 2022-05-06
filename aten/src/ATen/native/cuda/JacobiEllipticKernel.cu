#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Math.h>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/Math.cuh>
#include <ATen/native/cuda/jit_utils.h>

namespace at {
namespace native {
namespace {
const char jacobi_elliptic_k_cd_name[] = "jacobi_elliptic_k_cd";

void jacobi_elliptic_k_cd_kernel_cuda(TensorIteratorBase& iterator) {
#if AT_USE_JITERATOR()
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_elliptic_k_cd_cuda", [&]() {
        opmath_jitted_gpu_kernel_with_scalars<jacobi_elliptic_k_cd_name, scalar_t, scalar_t>(iterator, jacobi_elliptic_k_cd_string);
    });
#else
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_elliptic_k_cd_cuda", [&]() {
        gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t u, scalar_t k) -> scalar_t {
            return jacobi_elliptic_k_cd<scalar_t, true>(u, k);
        });
    });
#endif
} // jacobi_elliptic_k_cd_kernel_cuda

const char jacobi_elliptic_k_cn_name[] = "jacobi_elliptic_k_cn";

void jacobi_elliptic_k_cn_kernel_cuda(TensorIteratorBase& iterator) {
#if AT_USE_JITERATOR()
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_elliptic_k_cn_cuda", [&]() {
        opmath_jitted_gpu_kernel_with_scalars<jacobi_elliptic_k_cn_name, scalar_t, scalar_t>(iterator, jacobi_elliptic_k_cn_string);
    });
#else
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_elliptic_k_cn_cuda", [&]() {
        gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t u, scalar_t k) -> scalar_t {
            return jacobi_elliptic_k_cn<scalar_t, true>(u, k);
        });
    });
#endif
} // jacobi_elliptic_k_cn_kernel_cuda

const char jacobi_elliptic_k_cs_name[] = "jacobi_elliptic_k_cs";

void jacobi_elliptic_k_cs_kernel_cuda(TensorIteratorBase& iterator) {
#if AT_USE_JITERATOR()
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_elliptic_k_cs_cuda", [&]() {
        opmath_jitted_gpu_kernel_with_scalars<jacobi_elliptic_k_cs_name, scalar_t, scalar_t>(iterator, jacobi_elliptic_k_cs_string);
    });
#else
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_elliptic_k_cs_cuda", [&]() {
        gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t u, scalar_t k) -> scalar_t {
            return jacobi_elliptic_k_cs<scalar_t, true>(u, k);
        });
    });
#endif
} // jacobi_elliptic_k_cs_kernel_cuda

const char jacobi_elliptic_k_dc_name[] = "jacobi_elliptic_k_dc";

void jacobi_elliptic_k_dc_kernel_cuda(TensorIteratorBase& iterator) {
#if AT_USE_JITERATOR()
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_elliptic_k_dc_cuda", [&]() {
        opmath_jitted_gpu_kernel_with_scalars<jacobi_elliptic_k_dc_name, scalar_t, scalar_t>(iterator, jacobi_elliptic_k_dc_string);
    });
#else
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_elliptic_k_dc_cuda", [&]() {
        gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t u, scalar_t k) -> scalar_t {
            return jacobi_elliptic_k_dc<scalar_t, true>(u, k);
        });
    });
#endif
} // jacobi_elliptic_k_dc_kernel_cuda

const char jacobi_elliptic_k_dn_name[] = "jacobi_elliptic_k_dn";

void jacobi_elliptic_k_dn_kernel_cuda(TensorIteratorBase& iterator) {
#if AT_USE_JITERATOR()
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_elliptic_k_dn_cuda", [&]() {
        opmath_jitted_gpu_kernel_with_scalars<jacobi_elliptic_k_dn_name, scalar_t, scalar_t>(iterator, jacobi_elliptic_k_dn_string);
    });
#else
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_elliptic_k_dn_cuda", [&]() {
        gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t u, scalar_t k) -> scalar_t {
            return jacobi_elliptic_k_dn<scalar_t, true>(u, k);
        });
    });
#endif
} // jacobi_elliptic_k_dn_kernel_cuda

const char jacobi_elliptic_k_ds_name[] = "jacobi_elliptic_k_ds";

void jacobi_elliptic_k_ds_kernel_cuda(TensorIteratorBase& iterator) {
#if AT_USE_JITERATOR()
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_elliptic_k_ds_cuda", [&]() {
        opmath_jitted_gpu_kernel_with_scalars<jacobi_elliptic_k_ds_name, scalar_t, scalar_t>(iterator, jacobi_elliptic_k_ds_string);
    });
#else
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_elliptic_k_ds_cuda", [&]() {
        gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t u, scalar_t k) -> scalar_t {
            return jacobi_elliptic_k_ds<scalar_t, true>(u, k);
        });
    });
#endif
} // jacobi_elliptic_k_ds_kernel_cuda

const char jacobi_elliptic_k_nc_name[] = "jacobi_elliptic_k_nc";

void jacobi_elliptic_k_nc_kernel_cuda(TensorIteratorBase& iterator) {
#if AT_USE_JITERATOR()
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_elliptic_k_nc_cuda", [&]() {
        opmath_jitted_gpu_kernel_with_scalars<jacobi_elliptic_k_nc_name, scalar_t, scalar_t>(iterator, jacobi_elliptic_k_nc_string);
    });
#else
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_elliptic_k_nc_cuda", [&]() {
        gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t u, scalar_t k) -> scalar_t {
            return jacobi_elliptic_k_nc<scalar_t, true>(u, k);
        });
    });
#endif
} // jacobi_elliptic_k_nc_kernel_cuda

const char jacobi_elliptic_k_nd_name[] = "jacobi_elliptic_k_nd";

void jacobi_elliptic_k_nd_kernel_cuda(TensorIteratorBase& iterator) {
#if AT_USE_JITERATOR()
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_elliptic_k_nd_cuda", [&]() {
        opmath_jitted_gpu_kernel_with_scalars<jacobi_elliptic_k_nd_name, scalar_t, scalar_t>(iterator, jacobi_elliptic_k_nd_string);
    });
#else
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_elliptic_k_nd_cuda", [&]() {
        gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t u, scalar_t k) -> scalar_t {
            return jacobi_elliptic_k_nd<scalar_t, true>(u, k);
        });
    });
#endif
} // jacobi_elliptic_k_nd_kernel_cuda

const char jacobi_elliptic_k_ns_name[] = "jacobi_elliptic_k_ns";

void jacobi_elliptic_k_ns_kernel_cuda(TensorIteratorBase& iterator) {
#if AT_USE_JITERATOR()
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_elliptic_k_ns_cuda", [&]() {
        opmath_jitted_gpu_kernel_with_scalars<jacobi_elliptic_k_ns_name, scalar_t, scalar_t>(iterator, jacobi_elliptic_k_ns_string);
    });
#else
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_elliptic_k_ns_cuda", [&]() {
        gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t u, scalar_t k) -> scalar_t {
            return jacobi_elliptic_k_ns<scalar_t, true>(u, k);
        });
    });
#endif
} // jacobi_elliptic_k_ns_kernel_cuda

const char jacobi_elliptic_k_sc_name[] = "jacobi_elliptic_k_sc";

void jacobi_elliptic_k_sc_kernel_cuda(TensorIteratorBase& iterator) {
#if AT_USE_JITERATOR()
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_elliptic_k_sc_cuda", [&]() {
        opmath_jitted_gpu_kernel_with_scalars<jacobi_elliptic_k_sc_name, scalar_t, scalar_t>(iterator, jacobi_elliptic_k_sc_string);
    });
#else
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_elliptic_k_sc_cuda", [&]() {
        gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t u, scalar_t k) -> scalar_t {
            return jacobi_elliptic_k_sc<scalar_t, true>(u, k);
        });
    });
#endif
} // jacobi_elliptic_k_sc_kernel_cuda

const char jacobi_elliptic_k_sd_name[] = "jacobi_elliptic_k_sd";

void jacobi_elliptic_k_sd_kernel_cuda(TensorIteratorBase& iterator) {
#if AT_USE_JITERATOR()
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_elliptic_k_sd_cuda", [&]() {
        opmath_jitted_gpu_kernel_with_scalars<jacobi_elliptic_k_sd_name, scalar_t, scalar_t>(iterator, jacobi_elliptic_k_sd_string);
    });
#else
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_elliptic_k_sd_cuda", [&]() {
        gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t u, scalar_t k) -> scalar_t {
            return jacobi_elliptic_k_sd<scalar_t, true>(u, k);
        });
    });
#endif
} // jacobi_elliptic_k_sd_kernel_cuda

const char jacobi_elliptic_k_sn_name[] = "jacobi_elliptic_k_sn";

void jacobi_elliptic_k_sn_kernel_cuda(TensorIteratorBase& iterator) {
#if AT_USE_JITERATOR()
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_elliptic_k_sn_cuda", [&]() {
        opmath_jitted_gpu_kernel_with_scalars<jacobi_elliptic_k_sn_name, scalar_t, scalar_t>(iterator, jacobi_elliptic_k_sn_string);
    });
#else
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_elliptic_k_sn_cuda", [&]() {
        gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t u, scalar_t k) -> scalar_t {
            return jacobi_elliptic_k_sn<scalar_t, true>(u, k);
        });
    });
#endif
} // jacobi_elliptic_k_sn_kernel_cuda
} // anonymous

REGISTER_DISPATCH(jacobi_elliptic_k_cd_stub, &jacobi_elliptic_k_cd_kernel_cuda);
REGISTER_DISPATCH(jacobi_elliptic_k_cn_stub, &jacobi_elliptic_k_cn_kernel_cuda);
REGISTER_DISPATCH(jacobi_elliptic_k_cs_stub, &jacobi_elliptic_k_cs_kernel_cuda);
REGISTER_DISPATCH(jacobi_elliptic_k_dc_stub, &jacobi_elliptic_k_dc_kernel_cuda);
REGISTER_DISPATCH(jacobi_elliptic_k_dn_stub, &jacobi_elliptic_k_dn_kernel_cuda);
REGISTER_DISPATCH(jacobi_elliptic_k_ds_stub, &jacobi_elliptic_k_ds_kernel_cuda);
REGISTER_DISPATCH(jacobi_elliptic_k_nc_stub, &jacobi_elliptic_k_nc_kernel_cuda);
REGISTER_DISPATCH(jacobi_elliptic_k_nd_stub, &jacobi_elliptic_k_nd_kernel_cuda);
REGISTER_DISPATCH(jacobi_elliptic_k_ns_stub, &jacobi_elliptic_k_ns_kernel_cuda);
REGISTER_DISPATCH(jacobi_elliptic_k_sc_stub, &jacobi_elliptic_k_sc_kernel_cuda);
REGISTER_DISPATCH(jacobi_elliptic_k_sd_stub, &jacobi_elliptic_k_sd_kernel_cuda);
REGISTER_DISPATCH(jacobi_elliptic_k_sn_stub, &jacobi_elliptic_k_sn_kernel_cuda);
} // native
} // at
