#pragma once

#include <c10/xpu/XPUMacros.h>
#include <sycl/sycl.hpp>

namespace c10::xpu {

#define AT_FORALL_XPU_DEVICE_PROPERTIES(_)                                     \
  /* the device name of this SYCL device. */                                   \
  _(name)                                                                      \
                                                                               \
  /* the device type associated with the device. */                            \
  _(device_type)                                                               \
                                                                               \
  /* the vendor of this SYCL device. */                                        \
  _(vendor)                                                                    \
                                                                               \
  /* a backend-defined driver version as a std::string. */                     \
  _(driver_version)                                                            \
                                                                               \
  /* the SYCL version as a std::string in the form <major>.<minor> */          \
  _(version)                                                                   \
                                                                               \
  /* true if the SYCL device is available. Otherwise, return false. */         \
  _(is_available)                                                              \
                                                                               \
  /* the maximum size in bytes of the arguments that can be passed to a        \
   * kernel. */                                                                \
  _(max_parameter_size)                                                        \
                                                                               \
  /* the number of parallel compute units available to the device. */          \
  _(max_compute_units)                                                         \
                                                                               \
  /* the maximum dimensions that specify the global and local work-item IDs    \
   * used by the data parallel execution model. */                             \
  _(max_work_item_dimensions)                                                  \
                                                                               \
  /* the maximum number of workitems that are permitted in a work-group        \
   * executing a kernel on a single compute unit. */                           \
  _(max_work_group_size)                                                       \
                                                                               \
  /* the maximum number of subgroups in a work-group for any kernel executed   \
   * on the device. */                                                         \
  _(max_num_sub_groups)                                                        \
                                                                               \
  /* a std::vector of size_t containing the set of sub-group sizes  supported  \
   * by the device. */                                                         \
  _(sub_group_sizes)                                                           \
                                                                               \
  /* the maximum configured clock frequency of this SYCL device in MHz. */     \
  _(max_clock_frequency)                                                       \
                                                                               \
  /* the default compute device address space size specified as an unsigned    \
   * integer value in bits. Must return either 32 or 64. */                    \
  _(address_bits)                                                              \
                                                                               \
  /* the maximum size of memory object allocation in bytes. */                 \
  _(max_mem_alloc_size)                                                        \
                                                                               \
  /* the minimum value in bits of the largest supported SYCL built-in data     \
   * type if this SYCL device is not of device type                            \
   * sycl::info::device_type::custom. */                                       \
  _(mem_base_addr_align)                                                       \
                                                                               \
  /* a std::vector of info::fp_config describing the half/single/double        \
   * precision floating-point capability of this SYCL device. */               \
  _(half_fp_config)                                                            \
  _(single_fp_config)                                                          \
  _(double_fp_config)                                                          \
                                                                               \
  /* the size of global device memory in bytes. */                             \
  _(global_mem_size)                                                           \
                                                                               \
  /* the type of global memory cache supported. */                             \
  _(global_mem_cache_type)                                                     \
                                                                               \
  /* the size of global memory cache in bytes. */                              \
  _(global_mem_cache_size)                                                     \
                                                                               \
  /* the size of global memory cache line in bytes. */                         \
  _(global_mem_cache_line_size)                                                \
                                                                               \
  /* the type of local memory supported. */                                    \
  _(local_mem_type)                                                            \
                                                                               \
  /* the size of local memory arena in bytes. */                               \
  _(local_mem_size)                                                            \
                                                                               \
  /* the maximum number of sub-devices that can be created when this device is \
   * partitioned. */                                                           \
  _(partition_max_sub_devices)                                                 \
                                                                               \
  /* the resolution of device timer in nanoseconds. */                         \
  _(profiling_timer_resolution)                                                \
                                                                               \
  /* the preferred native vector width size for built-in scalar types that can \
   * be put into vectors. */                                                   \
  _(preferred_vector_width_char)                                               \
  _(preferred_vector_width_short)                                              \
  _(preferred_vector_width_int)                                                \
  _(preferred_vector_width_long)                                               \
  _(preferred_vector_width_float)                                              \
  _(preferred_vector_width_double)                                             \
  _(preferred_vector_width_half)                                               \
                                                                               \
  /* the native ISA vector width. The vector width is defined as the number of \
   * scalar elements that can be stored in the vector. */                      \
  _(native_vector_width_char)                                                  \
  _(native_vector_width_short)                                                 \
  _(native_vector_width_int)                                                   \
  _(native_vector_width_long)                                                  \
  _(native_vector_width_float)                                                 \
  _(native_vector_width_double)                                                \
  _(native_vector_width_half)

#define AT_FORALL_XPU_EXT_DEVICE_PROPERTIES(_)           \
  /* the number of EUs associated with the Intel GPU. */ \
  _(gpu_eu_count, 512)                                   \
                                                         \
  /* the number of EUs in a subslice. */                 \
  _(gpu_eu_count_per_subslice, 8)                        \
                                                         \
  /* the simd width of EU of GPU. */                     \
  _(gpu_eu_simd_width, 8)                                \
                                                         \
  /* the number of hardware threads per EU of GPU. */    \
  _(gpu_hw_threads_per_eu, 8)

#define AT_FORALL_XPU_DEVICE_ASPECT(_)                  \
  /* sycl::half is supported on device. */              \
  _(fp16)                                               \
                                                        \
  /* double is supported on device. */                  \
  _(fp64)                                               \
                                                        \
  /* 64-bit atomic operation is supported on device. */ \
  _(atomic64)

#define AT_FORALL_XPU_EXP_CL_ASPECT(_)                                         \
  /* conversion between single-precision 32-bit floating-point values and      \
   * 16-bit bfloat16 values is supported on device. */                         \
  _(bfloat16_conversions)                                                      \
                                                                               \
  /* specialized hardware to compute MMA is supported on device. */            \
  _(subgroup_matrix_multiply_accumulate)                                       \
                                                                               \
  /* specialized hardware to compute MMA for 32-bit floating-point is          \
   * supported on device. */                                                   \
  _(subgroup_matrix_multiply_accumulate_tensor_float32)                        \
                                                                               \
  /* block read operations for efficient matrix multiplication is supported on \
   * device. */                                                                \
  _(subgroup_2d_block_io)

#define AT_FORALL_XPU_EXP_DEVICE_PROPERTIES(_)       \
  /* the device architecture of this SYCL device. */ \
  _(architecture)

#define _DEFINE_SYCL_PROP(ns, property, member) \
  ns::property::return_type member;

#define DEFINE_DEVICE_PROP(property) \
  _DEFINE_SYCL_PROP(sycl::info::device, property, property)

#define DEFINE_PLATFORM_PROP(property, member) \
  _DEFINE_SYCL_PROP(sycl::info::platform, property, member)

#define DEFINE_EXT_DEVICE_PROP(property, ...) \
  _DEFINE_SYCL_PROP(sycl::ext::intel::info::device, property, property)

#define DEFINE_DEVICE_ASPECT(member) bool has_##member;

#define DEFINE_EXP_DEVICE_PROP(property) \
  _DEFINE_SYCL_PROP(                     \
      sycl::ext::oneapi::experimental::info::device, property, property)

struct C10_XPU_API DeviceProp {
  AT_FORALL_XPU_DEVICE_PROPERTIES(DEFINE_DEVICE_PROP);

  // the platform name.
  DEFINE_PLATFORM_PROP(name, platform_name);

  AT_FORALL_XPU_EXT_DEVICE_PROPERTIES(DEFINE_EXT_DEVICE_PROP);

  AT_FORALL_XPU_DEVICE_ASPECT(DEFINE_DEVICE_ASPECT);

  AT_FORALL_XPU_EXP_CL_ASPECT(DEFINE_DEVICE_ASPECT);

#if SYCL_COMPILER_VERSION >= 20250000
  AT_FORALL_XPU_EXP_DEVICE_PROPERTIES(DEFINE_EXP_DEVICE_PROP);
#endif
};

#undef _DEFINE_SYCL_PROP
#undef DEFINE_DEVICE_PROP
#undef DEFINE_PLATFORM_PROP
#undef DEFINE_EXT_DEVICE_PROP
#undef DEFINE_DEVICE_ASPECT
#undef DEFINE_EXP_DEVICE_PROP

} // namespace c10::xpu
