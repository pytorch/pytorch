#pragma once

#include <c10/xpu/XPUMacros.h>
#include <sycl/sycl.hpp>

namespace c10::xpu {

struct C10_XPU_API DeviceProp {
#define SYCL_PROP(ns, property, member) ns::property::return_type member

#define SYCL_DEVICE_PROP(property, member) \
  SYCL_PROP(sycl::info::device, property, member)

#define SYCL_PLATFORM_PROP(property, member) \
  SYCL_PROP(sycl::info::platform, property, member)

#define SYCL_EXT_DEVICE_PROP(property, member) \
  SYCL_PROP(sycl::ext::intel::info::device, property, member)

  // Returns the device name of this SYCL device.
  SYCL_DEVICE_PROP(name, device_name);

  // Returns the device type associated with the device.
  SYCL_DEVICE_PROP(device_type, device_type);

  // Returns the platform name.
  SYCL_PLATFORM_PROP(name, platform_name);

  // Returns the vendor of this SYCL device.
  SYCL_DEVICE_PROP(vendor, vendor);

  // Returns a backend-defined driver version as a std::string.
  SYCL_DEVICE_PROP(driver_version, driver_version);

  // Returns the SYCL version as a std::string in the form:
  // <major_version>.<minor_version>
  SYCL_DEVICE_PROP(version, version);

  // Returns true if the SYCL device is available. Otherwise, return false.
  SYCL_DEVICE_PROP(is_available, is_available);

  // Returns the maximum size in bytes of the arguments that can be passed to a
  // kernel.
  SYCL_DEVICE_PROP(max_parameter_size, max_param_size);

  // Returns the number of parallel compute units available to the device.
  SYCL_DEVICE_PROP(max_compute_units, max_compute_units);

  // Returns the maximum dimensions that specify the global and local work-item
  // IDs used by the data parallel execution model.
  SYCL_DEVICE_PROP(max_work_item_dimensions, max_work_item_dims);

  // Returns the maximum number of workitems that are permitted in a work-group
  // executing a kernel on a single compute unit.
  SYCL_DEVICE_PROP(max_work_group_size, max_work_group_size);

  // Returns the maximum number of subgroups in a work-group for any kernel
  // executed on the device.
  SYCL_DEVICE_PROP(max_num_sub_groups, max_num_sub_groups);

  // Returns a std::vector of size_t containing the set of sub-group sizes
  // supported by the device.
  SYCL_DEVICE_PROP(sub_group_sizes, sub_group_sizes);

  // Returns the maximum configured clock frequency of this SYCL device in MHz.
  SYCL_DEVICE_PROP(max_clock_frequency, max_clock_freq);

  // Returns the default compute device address space size specified as an
  // unsigned integer value in bits. Must return either 32 or 64.
  SYCL_DEVICE_PROP(address_bits, address_bits);

  // Returns the maximum size of memory object allocation in bytes.
  SYCL_DEVICE_PROP(max_mem_alloc_size, max_mem_alloc_size);

  // Returns the minimum value in bits of the largest supported SYCL built-in
  // data type if this SYCL device is not of device type
  // sycl::info::device_type::custom.
  SYCL_DEVICE_PROP(mem_base_addr_align, mem_base_addr_align);

  // Returns a std::vector of info::fp_config describing the half precision
  // floating-point capability of this SYCL device.
  SYCL_DEVICE_PROP(half_fp_config, half_fp_config);

  // Returns a std::vector of info::fp_config describing the single precision
  // floating-point capability of this SYCL device.
  SYCL_DEVICE_PROP(single_fp_config, single_fp_config);

  // Returns a std::vector of info::fp_config describing the double precision
  // floating-point capability of this SYCL device.
  SYCL_DEVICE_PROP(double_fp_config, double_fp_config);

  // Returns the size of global device memory in bytes.
  SYCL_DEVICE_PROP(global_mem_size, global_mem_size);

  // Returns the type of global memory cache supported.
  SYCL_DEVICE_PROP(global_mem_cache_type, global_mem_cache_type);

  // Returns the size of global memory cache in bytes.
  SYCL_DEVICE_PROP(global_mem_cache_size, global_mem_cache_size);

  // Returns the size of global memory cache line in bytes.
  SYCL_DEVICE_PROP(global_mem_cache_line_size, global_mem_cache_line_size);

  // Returns the type of local memory supported.
  SYCL_DEVICE_PROP(local_mem_type, local_mem_type);

  // Returns the size of local memory arena in bytes.
  SYCL_DEVICE_PROP(local_mem_size, local_mem_size);

  // Returns the maximum number of sub-devices that can be created when this
  // device is partitioned.
  SYCL_DEVICE_PROP(partition_max_sub_devices, max_sub_devices);

  // Returns the resolution of device timer in nanoseconds.
  SYCL_DEVICE_PROP(profiling_timer_resolution, profiling_resolution);

  // Returns the preferred native vector width size for built-in scalar types
  // that can be put into vectors.
  SYCL_DEVICE_PROP(preferred_vector_width_char, pref_vec_width_char);
  SYCL_DEVICE_PROP(preferred_vector_width_short, pref_vec_width_short);
  SYCL_DEVICE_PROP(preferred_vector_width_int, pref_vec_width_int);
  SYCL_DEVICE_PROP(preferred_vector_width_long, pref_vec_width_long);
  SYCL_DEVICE_PROP(preferred_vector_width_float, pref_vec_width_float);
  SYCL_DEVICE_PROP(preferred_vector_width_double, pref_vec_width_double);
  SYCL_DEVICE_PROP(preferred_vector_width_half, pref_vec_width_half);

  // Returns the native ISA vector width. The vector width is defined as the
  // number of scalar elements that can be stored in the vector.
  SYCL_DEVICE_PROP(native_vector_width_char, native_vec_width_char);
  SYCL_DEVICE_PROP(native_vector_width_short, native_vec_width_short);
  SYCL_DEVICE_PROP(native_vector_width_int, native_vec_width_int);
  SYCL_DEVICE_PROP(native_vector_width_long, native_vec_width_long);
  SYCL_DEVICE_PROP(native_vector_width_float, native_vec_width_float);
  SYCL_DEVICE_PROP(native_vector_width_double, native_vec_width_double);
  SYCL_DEVICE_PROP(native_vector_width_half, native_vec_width_half);

  // Returns the number of EUs associated with the Intel GPU.
  SYCL_EXT_DEVICE_PROP(gpu_eu_count, gpu_eu_count);

  // Returns the number of EUs in a subslice.
  SYCL_EXT_DEVICE_PROP(gpu_eu_count_per_subslice, gpu_eu_count_per_subslice);

  // Returns the simd width of EU of GPU.
  SYCL_EXT_DEVICE_PROP(gpu_eu_simd_width, gpu_eu_simd_width);

  // Returns the number of hardware threads per EU of GPU.
  SYCL_EXT_DEVICE_PROP(gpu_hw_threads_per_eu, gpu_hw_threads_per_eu);

#undef SYCL_DEVICE_PROP
#undef SYCL_PLATFORM_PROP
#undef SYCL_EXT_DEVICE_PROP
};

} // namespace c10::xpu
