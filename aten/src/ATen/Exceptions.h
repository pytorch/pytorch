#pragma once

#include <string>
#include <sstream>

#include <ATen/Tensor.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

namespace at {

/// Get name of the device the tensor is actually on
inline std::string torch_tensor_device_name(const at::Tensor &ten){
  return c10::DeviceTypeName(ten.device().type());
}

/// Get name of the device the tensor is actually on
inline std::string torch_tensor_device_name(const c10::optional<at::Tensor> &ten){
  if(ten.has_value()){
    return c10::DeviceTypeName(ten->device().type());
  } else {
    return "No device: optional tensor unused.";
  }
}

/// Overloaded function to determine if a tensor is on a CPU
inline bool torch_tensor_on_cpu_check(const at::Tensor& ten) {
  return ten.is_cpu();
}

/// Overloaded function to determine if a tensor is on a CPU
///
/// If the optional tensor isn't present then we claim it is on
/// the GPU. This has the effect of making the optional tensor
/// compatible with any combination of devices.
inline bool torch_tensor_on_cpu_check(
    const c10::optional<at::Tensor>& ten) {
  return !ten.has_value() || torch_tensor_on_cpu_check(*ten);
}

/// Overloaded function to determine if a tensor is on a GPU
inline bool torch_tensor_on_cuda_gpu_check(const at::Tensor& ten) {
  return ten.is_cuda();
}

/// Overloaded function to determine if a tensor is on a GPU
///
/// If the optional tensor isn't present then we claim it is on
/// the GPU. This has the effect of making the optional tensor
/// compatible with any combination of devices.
inline bool torch_tensor_on_cuda_gpu_check(
    const c10::optional<at::Tensor>& ten) {
  return !ten.has_value() || torch_tensor_on_cuda_gpu_check(*ten);
}

namespace detail {

/// Check if a tensor is contiguous
#define TORCH_CHECK_TENSOR_CONTIGUOUS(x) \
  TORCH_CHECK((x).is_contiguous(), #x " must be contiguous at ", __FILE__, ":", __LINE__)

/// Raise an exception if tensors are not on the same device
#define TORCH_CHECK_TENSORS_ON_SAME_DEVICE(X...)                                       \
  at::detail::tensors_on_same_device_macro_expansion<at::detail::CheckForDevice::NONE> \
  (0, __FILE__, __LINE__, #X, X)

/// Raise an exception if tensors are not on the same GPU
#define TORCH_CHECK_TENSORS_ON_SAME_CUDA_GPU(X...)                                     \
  at::detail::tensors_on_same_device_macro_expansion<at::detail::CheckForDevice::CUDA> \
  (0, __FILE__, __LINE__, #X, X)

/// Raise an exception if tensors are not on the same CPU
#define TORCH_CHECK_TENSORS_ON_SAME_CPU(X...)                                         \
  at::detail::tensors_on_same_device_macro_expansion<at::detail::CheckForDevice::CPU> \
  (0, __FILE__, __LINE__, #X, X)

/// Used to determine which of the two tensors being compared at any one time
/// provides an authoritative declaration of where the tensors should be
/// located. That is, if one of the tensors is optional and not present, we
/// choose the other one as indicative of the device
enum class SameDeviceStatus {
  DIFFERENT_DEVICES, // Tensors on different devices
  SAME_DEVICE_LEFT, // Tensors on same device and left-argument tensor was
                    // non-optional
  SAME_DEVICE_RIGHT, // Tensors on same device and right-argument tensor was
                     // non-optional
};

/// Used to specify whether we check if the tensors are on a specific device type
enum class CheckForDevice {
  NONE, //< Don't perform such a device-specific check
  CUDA, //< Ensure tensors are on CUDA
  CPU, //< Ensure tensors are on CPU
};

/// Determine whether two normal tensors are on the same device
inline SameDeviceStatus on_same_device(
    const at::Tensor& ten1,
    const at::Tensor& ten2) {
  if (ten1.get_device() == ten2.get_device()) {
    return SameDeviceStatus::SAME_DEVICE_RIGHT;
  } else {
    return SameDeviceStatus::DIFFERENT_DEVICES;
  }
}

/// Determine whether a tensor and an optional tensor are on the same device
inline SameDeviceStatus on_same_device(
    const at::Tensor& ten1,
    const c10::optional<at::Tensor>& ten2) {
  if (!ten2.has_value() || ten1.get_device() == ten2->get_device()) {
    // Either right tensor was on same device or right tensor was not present
    // either way, the left tensor is authoritative.
    return SameDeviceStatus::SAME_DEVICE_LEFT;
  } else {
    return SameDeviceStatus::DIFFERENT_DEVICES;
  }
}

/// Determine whether an optional tensor and a tensor are on the same device
inline SameDeviceStatus on_same_device(
    const c10::optional<at::Tensor>& ten1,
    const at::Tensor& ten2) {
  if (!ten1.has_value() || ten1->get_device() == ten2.get_device()) {
    // Either left tensor was on same device or left tensor was not present
    // either way, the right tensor is authoritative.
    return SameDeviceStatus::SAME_DEVICE_RIGHT;
  } else {
    return SameDeviceStatus::DIFFERENT_DEVICES;
  }
}

/// Determine whether two optional tensors are on the same device
inline SameDeviceStatus on_same_device(
    const c10::optional<at::Tensor>& ten1,
    const c10::optional<at::Tensor>& ten2) {
  if (ten1.has_value() && ten2.has_value()) {
    // Both tensors present, so we compare them as normal
    if (ten1->get_device() == ten2->get_device()) {
      return SameDeviceStatus::SAME_DEVICE_RIGHT;
    } else {
      return SameDeviceStatus::DIFFERENT_DEVICES;
    }
  } else if (ten1.has_value()) {
    // Only the left tensor was present, so that's authoritative
    return SameDeviceStatus::SAME_DEVICE_LEFT;
  } else {
    // Only the right tensor was present, so that's authoritative
    return SameDeviceStatus::SAME_DEVICE_RIGHT;
  }
}

/// Break apart argument string to get the argument at position `pos`
///
/// The arguments passed to `TENSORS_ON_SAME_DEVICE` are in a comma-delimited
/// string here, thanks to the `#X` in the macros. We break that string apart
/// to extract the name of the mislocated tensor and will use that name to
/// make a helpful error message
inline std::string break_args(const std::string& arg_names, const int pos) {
  std::stringstream s_stream(arg_names); // create string stream from the string
  for (int i = 0; s_stream.good(); i++) {
    std::string substr;
    getline(s_stream, substr, ','); // get first string delimited by comma
    if (i == pos) {
      return substr;
    }
  }
  throw std::runtime_error("Something horrible happened!");
}

/// Specialization to end the recursive parsing of arguments
template <CheckForDevice check_for_device, typename TensorType>
void assert_that_tensor_is_on_correct_device(
    const int depth,
    const std::string &filename,
    const int line_number,
    const std::string &name_list,
    const TensorType& ten
) {
  // Determine if tensors are on a particular kind of device
  switch(check_for_device){
    case CheckForDevice::CPU:
      TORCH_CHECK(
        torch_tensor_on_cpu_check(ten),
        "On '" + filename + ":" + std::to_string(line_number) + "' the tensor '" + break_args(name_list, depth) + " is not on a CPU! It is actually on " + torch_tensor_device_name(ten)
      );
      break;
    case CheckForDevice::CUDA:
      TORCH_CHECK(
        torch_tensor_on_cuda_gpu_check(ten),
        "On '" + filename + ":" + std::to_string(line_number) + "' the tensor '" + break_args(name_list, depth) + " is not on a CUDA device! It is actually on " + torch_tensor_device_name(ten)
      );
      break;
    case CheckForDevice::NONE:
      break;
    default:
      break; // Should never reach this point
  }
}

/// Specialization to end the recursive parsing of arguments
template <CheckForDevice check_for_device, typename Last>
void tensors_on_same_device_macro_expansion(
    const int depth,
    const std::string &filename,
    const int line_number,
    const std::string &name_list,
    const Last& last) {
  assert_that_tensor_is_on_correct_device<check_for_device>(
    depth, filename, line_number, name_list, last
  );

  // At this point we know that all tensors are on the correct device
  // and that they are on the same device.
}

/// Parse arguments recursively comparing subsequent tensors (rearranging as
/// necessary to handle optional tensors) until we prove that all tensors
/// are on the same device or identify which one is not.
///
/// Return value indicates which tensor was on the wrong device. This value
/// never makes it to the user.
template <CheckForDevice check_for_device, typename First, typename Second, typename... Rest>
void tensors_on_same_device_macro_expansion(
    const int depth, // Depth in the recursion
    const std::string &filename, // Name of the source code file containing check
    const int
        &line_number, // Line number in the source code file containing check
    const std::string
        &name_list, // Comma-delimited string of arguments passed to the check
    const First& f, // First argument to the check
    const Second& s, // Second argument to the check
    const Rest&... rest // All other arguments to the check
) {
  assert_that_tensor_is_on_correct_device<check_for_device>(depth, filename, line_number, name_list, f);

  // Determine if tensors are on the same device. If so, recurse downwards and drop
  // an argument. Keep whichever argument contained a non-optional tensor.
  const auto sdstatus = on_same_device(f, s);
  if (sdstatus == SameDeviceStatus::SAME_DEVICE_LEFT) {
    tensors_on_same_device_macro_expansion<check_for_device, First, Rest...>(
        depth + 1, filename, line_number, name_list, f, rest...);
  } else if (sdstatus == SameDeviceStatus::SAME_DEVICE_RIGHT) {
    tensors_on_same_device_macro_expansion<check_for_device, Second, Rest...>(
        depth + 1, filename, line_number, name_list, s, rest...);
  } else {
    // Tensors were on different devices, make a note of which tensor was in
    // the wrong place.
    TORCH_CHECK(
      false,
      "On '" + filename + ":" + std::to_string(line_number) + "' the tensor '"
              + break_args(name_list, depth + 1)
              + "' is on a different device than all the tensors before it."
    );
  }
}

}

}
