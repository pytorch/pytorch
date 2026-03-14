#pragma once

// Debugging utility for capturing AOTI kernel call inputs.
//
// Wrap any call_* kernel invocation with AOTI_CAPTURE_CALL to save all tensor
// and scalar arguments into a single .pt file per call, loadable via
// torch.load(). Infrastructure args (stream, kernels struct, cubin_dir) are
// automatically skipped.
//
// This header uses only the stable C ABI (shim.h) so it can be compiled into
// standalone AOTI .so builds without libtorch.
//
// Activation (env vars):
//   AOTI_KERNEL_CAPTURE_DIR=/path   — enable capture, set output directory
//   AOTI_KERNEL_CAPTURE_FILTER=name — optional: only capture matching kernels
//
// Usage:
//   #include <torch/csrc/inductor/aoti_runtime/kernel_capture.h>
//
//   // For call_* kernel wrappers (variadic args):
//   AOTI_CAPTURE_CALL(call_my_kernel, tensor_arg, scalar_arg, ...);
//
//   // With explicit provenance tag (e.g. debug handle):
//   AOTI_CAPTURE_CALL_TAG("4", call_my_kernel, tensor_arg, scalar_arg, ...);
//
//   // For proxy executor calls (flattened arrays):
//   AOTI_CAPTURE_PROXY(proxy_executor, extern_node_index, num_ints,
//       int_args, num_tensors, tensor_args);
//
//   // With explicit provenance tag:
//   AOTI_CAPTURE_PROXY_TAG("4", proxy_executor, extern_node_index,
//       num_ints, int_args, num_tensors, tensor_args);

#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

namespace aoti_kernel_capture {

// --- Type traits for argument classification ---

// Tensor-like: anything implicitly convertible to AtenTensorHandle
// (RAIIAtenTensorHandle, ConstantHandle, etc.)
template <typename T, typename = void>
struct is_tensor_like : std::false_type {};
template <typename T>
struct is_tensor_like<
    T,
    std::void_t<decltype(static_cast<AtenTensorHandle>(std::declval<const T&>()))>>
    : std::true_type {};

// Numeric scalar types we want to capture.
template <typename T>
struct is_int_scalar
    : std::bool_constant<
          std::is_same_v<std::decay_t<T>, int64_t> ||
          std::is_same_v<std::decay_t<T>, int32_t> ||
          std::is_same_v<std::decay_t<T>, bool>> {};

template <typename T>
struct is_float_scalar
    : std::bool_constant<
          std::is_same_v<std::decay_t<T>, double> ||
          std::is_same_v<std::decay_t<T>, float>> {};

// --- Capture argument accumulator (C ABI compatible) ---

struct CaptureArgs {
  std::vector<AtenTensorHandle> tensor_handles;
  std::vector<int32_t> tensor_positions;
  std::vector<int64_t> int_values;
  std::vector<int32_t> int_positions;
  std::vector<double> float_values;
  std::vector<int32_t> float_positions;
  int32_t next_pos = 0;
};

// Tensor-like args — record the handle and its position.
template <typename T>
auto maybe_append(CaptureArgs& args, const T& val)
    -> std::enable_if_t<is_tensor_like<T>::value> {
  AtenTensorHandle handle = val;
  args.tensor_handles.push_back(handle);
  args.tensor_positions.push_back(args.next_pos++);
}

// Integer scalar args.
template <typename T>
auto maybe_append(CaptureArgs& args, const T& val)
    -> std::enable_if_t<is_int_scalar<T>::value> {
  args.int_values.push_back(static_cast<int64_t>(val));
  args.int_positions.push_back(args.next_pos++);
}

// Float scalar args.
template <typename T>
auto maybe_append(CaptureArgs& args, const T& val)
    -> std::enable_if_t<is_float_scalar<T>::value> {
  args.float_values.push_back(static_cast<double>(val));
  args.float_positions.push_back(args.next_pos++);
}

// Infrastructure args (stream, kernels struct, cubin_dir, etc.) — skip.
template <typename T>
auto maybe_append(CaptureArgs&, const T&)
    -> std::enable_if_t<
        !is_tensor_like<T>::value &&
        !is_int_scalar<T>::value &&
        !is_float_scalar<T>::value> {}

// --- Request-scoped capture ID ---

// Globally unique request ID (atomic for cross-thread uniqueness).
inline int next_request_id() {
  static std::atomic<int> id{0};
  return id++;
}

// Per-thread request ID (thread_local avoids races between concurrent requests).
inline int& current_request_id() {
  static thread_local int req_id{0};
  return req_id;
}

// Call at the start of each forward pass to begin a new capture request.
inline void begin_capture_request() {
  current_request_id() = next_request_id();
}

// --- Shared save logic ---

inline const char* capture_dir() {
  static const char* dir = std::getenv("AOTI_KERNEL_CAPTURE_DIR");
  return dir;
}

inline const char* capture_filter() {
  static const char* filter = std::getenv("AOTI_KERNEL_CAPTURE_FILTER");
  return filter;
}

inline bool should_capture(const std::string& kernel_name) {
  if (!capture_dir()) {
    return false;
  }
  const char* filter = capture_filter();
  return !filter || kernel_name.find(filter) != std::string::npos;
}

inline void save_capture(
    const std::string& kernel_name,
    const std::string& tag,
    const CaptureArgs& args) {
  int req_id = current_request_id();

  std::string filepath = std::string(capture_dir()) + "/" + kernel_name + "_" +
      tag + "_req" + std::to_string(req_id) + ".pt";
  std::filesystem::create_directories(
      std::filesystem::path(filepath).parent_path());

  aoti_torch_save_kernel_capture(
      filepath.c_str(),
      kernel_name.c_str(),
      tag.c_str(),
      static_cast<int32_t>(args.tensor_handles.size()),
      args.tensor_handles.data(),
      args.tensor_positions.data(),
      static_cast<int32_t>(args.int_values.size()),
      args.int_values.data(),
      args.int_positions.data(),
      static_cast<int32_t>(args.float_values.size()),
      args.float_values.data(),
      args.float_positions.data());
}

// --- Capture for call_* kernel wrappers (variadic individual args) ---

template <typename... Args>
void maybe_capture(
    const std::string& kernel_name,
    const std::string& tag,
    const Args&... args) {
  if (!should_capture(kernel_name)) {
    return;
  }

  CaptureArgs capture_args;
  (maybe_append(capture_args, args), ...);
  save_capture(kernel_name, tag, capture_args);
}

// --- Capture + call for proxy executor (flattened arrays) ---

inline void capture_and_call_proxy_impl(
    const std::string& tag,
    AOTIProxyExecutorHandle proxy_executor,
    int extern_node_index,
    int num_ints,
    int64_t* flatten_int_args,
    int num_tensors,
    AtenTensorHandle* flatten_tensor_args) {
  std::string name = "proxy_executor_" + std::to_string(extern_node_index);
  if (should_capture(name)) {
    CaptureArgs capture_args;
    for (int i = 0; i < num_ints; ++i) {
      capture_args.int_values.push_back(flatten_int_args[i]);
      capture_args.int_positions.push_back(capture_args.next_pos++);
    }
    for (int i = 0; i < num_tensors; ++i) {
      capture_args.tensor_handles.push_back(flatten_tensor_args[i]);
      capture_args.tensor_positions.push_back(capture_args.next_pos++);
    }
    save_capture(name, tag, capture_args);
  }
  aoti_torch_proxy_executor_call_function(
      proxy_executor, extern_node_index, num_ints, flatten_int_args,
      num_tensors, flatten_tensor_args);
}

} // namespace aoti_kernel_capture

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define AOTI_STRINGIFY_(x) #x
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define AOTI_STRINGIFY(x) AOTI_STRINGIFY_(x)

// Capture + call with explicit provenance tag (e.g. debug handle "4").
// Note: __VA_ARGS__ is evaluated twice (once for capture, once for the call).
// Callers must ensure arguments have no side effects.
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define AOTI_CAPTURE_CALL_TAG(tag, func, ...)                            \
  do {                                                                   \
    ::aoti_kernel_capture::maybe_capture(#func, tag, __VA_ARGS__);       \
    func(__VA_ARGS__);                                                   \
  } while (0)

// Capture + call using __LINE__ as the default tag.
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define AOTI_CAPTURE_CALL(func, ...)                                     \
  AOTI_CAPTURE_CALL_TAG(AOTI_STRINGIFY(__LINE__), func, __VA_ARGS__)

// Proxy executor capture + call using __LINE__ as the default tag.
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define AOTI_CAPTURE_PROXY(...)                                          \
  ::aoti_kernel_capture::capture_and_call_proxy_impl(                    \
      AOTI_STRINGIFY(__LINE__), __VA_ARGS__)

// Proxy executor capture + call with explicit provenance tag.
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define AOTI_CAPTURE_PROXY_TAG(tag, ...)                                 \
  ::aoti_kernel_capture::capture_and_call_proxy_impl(tag, __VA_ARGS__)
