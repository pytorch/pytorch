#pragma once

#include <string>

#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <torch/csrc/inductor/cpp_wrapper/common.h>
#if defined(USE_XPU)
#include <torch/csrc/inductor/cpp_wrapper/device_internal/xpu.h>
#else
#include <torch/csrc/inductor/cpp_wrapper/device_internal/cuda.h>
#endif

struct LazyKernelCompileResult {
  std::string cubin_path;
  std::string mangled_name;
  int num_warps;
  int shared_mem;
  int xblock;
  int yblock;
  int zblock;
  int r0block;
  int rsplit;
  int rsplit_size;
  int config_index;
  int global_scratch;
  int profile_scratch;
};

static PyObject* (*_THPVariable_Wrap)(const at::TensorBase&) = nullptr;
static int32_t (*_THPUtils_unpackInt)(PyObject*) = nullptr;

// Cached module and function references
static PyObject* triton_lazy_compile_module = nullptr;
static PyObject* start_kernel_compile = nullptr;
static PyObject* run_triton_kernel_with_autotune = nullptr;

// Per-module dict for pending kernel compile results (avoids global state
// collisions when multiple compiled modules produce kernels with the same
// name).
static PyObject* _module_pending_kernels = nullptr;

static inline void loadLazyCompileFuncs() {
  if (triton_lazy_compile_module == nullptr) {
    triton_lazy_compile_module =
        PyImport_ImportModule("torch._inductor.runtime.triton_lazy_compile");
    AOTI_TORCH_CHECK(
        triton_lazy_compile_module, "Failed to import triton_lazy_compile");

    start_kernel_compile = PyObject_GetAttrString(
        triton_lazy_compile_module, "start_kernel_compile");
    AOTI_TORCH_CHECK(
        start_kernel_compile, "Failed to get start_kernel_compile");

    run_triton_kernel_with_autotune = PyObject_GetAttrString(
        triton_lazy_compile_module, "run_triton_kernel_with_autotune");
    AOTI_TORCH_CHECK(
        run_triton_kernel_with_autotune,
        "Failed to get run_triton_kernel_with_autotune");

    RAIIPyObject guards_mod = PyImport_ImportModule("torch._C._dynamo.guards");
    AOTI_TORCH_CHECK(guards_mod, "Failed to import torch._C._dynamo.guards");

    RAIIPyObject wrap_addr =
        PyObject_GetAttrString(guards_mod, "_torchinductor_thp_variable_wrap");
    AOTI_TORCH_CHECK(
        wrap_addr, "Failed to get _torchinductor_thp_variable_wrap");
    _THPVariable_Wrap = reinterpret_cast<decltype(_THPVariable_Wrap)>(
        PyLong_AsVoidPtr(wrap_addr));
    AOTI_TORCH_CHECK(_THPVariable_Wrap, "THPVariable_Wrap not resolved");

    RAIIPyObject unpack_addr = PyObject_GetAttrString(
        guards_mod, "_torchinductor_thputils_unpack_int");
    AOTI_TORCH_CHECK(
        unpack_addr, "Failed to get _torchinductor_thputils_unpack_int");
    _THPUtils_unpackInt = reinterpret_cast<decltype(_THPUtils_unpackInt)>(
        PyLong_AsVoidPtr(unpack_addr));
    AOTI_TORCH_CHECK(_THPUtils_unpackInt, "THPUtils_unpackInt not resolved");
  }
}

static inline std::string getStringAttr(PyObject* obj, const char* attr) {
  RAIIPyObject val = PyObject_GetAttrString(obj, attr);
  AOTI_TORCH_CHECK(val, "Failed to get attribute");
  return PyUnicode_AsUTF8(val);
}

static inline int getIntAttr(PyObject* obj, const char* attr) {
  RAIIPyObject val = PyObject_GetAttrString(obj, attr);
  AOTI_TORCH_CHECK(val, "Failed to get attribute");
  return _THPUtils_unpackInt(val);
}

static inline int getOptionalIntAttr(
    PyObject* obj,
    const char* attr,
    int sentinel = -1) {
  RAIIPyObject val = PyObject_GetAttrString(obj, attr);
  AOTI_TORCH_CHECK(val, "Failed to get attribute");
  return (val.get() != Py_None) ? _THPUtils_unpackInt(val) : sentinel;
}

static inline LazyKernelCompileResult extractCompileResult(PyObject* result) {
  LazyKernelCompileResult compile_result;
  compile_result.cubin_path = getStringAttr(result, "cubin_path");
  compile_result.mangled_name = getStringAttr(result, "mangled_name");
  compile_result.num_warps = getIntAttr(result, "num_warps");
  compile_result.shared_mem = getIntAttr(result, "shared_mem");
  compile_result.xblock = getIntAttr(result, "xblock");
  compile_result.yblock = getIntAttr(result, "yblock");
  compile_result.zblock = getIntAttr(result, "zblock");
  compile_result.r0block = getIntAttr(result, "r0block");
  compile_result.rsplit = getIntAttr(result, "rsplit");
  compile_result.rsplit_size = getIntAttr(result, "rsplit_size");
  compile_result.config_index = getOptionalIntAttr(result, "config_index");
  compile_result.global_scratch = getOptionalIntAttr(result, "global_scratch");
  compile_result.profile_scratch =
      getOptionalIntAttr(result, "profile_scratch");
  return compile_result;
}

template <typename T>
static inline PyObject* convertArgToPython(const T& arg) {
  using DecayedT = std::decay_t<T>;
  if constexpr (std::is_same_v<DecayedT, AtenTensorHandle>) {
    at::Tensor* tensor_ptr =
        torch::aot_inductor::tensor_handle_to_tensor_pointer(arg);
    return _THPVariable_Wrap(*tensor_ptr);
  } else if constexpr (std::is_same_v<
                           DecayedT,
                           torch::aot_inductor::RAIIAtenTensorHandle>) {
    at::Tensor* tensor_ptr =
        torch::aot_inductor::tensor_handle_to_tensor_pointer(arg.get());
    return _THPVariable_Wrap(*tensor_ptr);
  } else if constexpr (std::is_same_v<DecayedT, bool>) {
    PyObject* py_arg = arg ? Py_True : Py_False;
    Py_INCREF(py_arg);
    return py_arg;
  } else if constexpr (std::is_integral_v<DecayedT>) {
    return PyLong_FromLongLong(static_cast<long long>(arg));
  } else if constexpr (std::is_floating_point_v<DecayedT>) {
    return PyFloat_FromDouble(static_cast<double>(arg));
  } else {
    AOTI_TORCH_CHECK(false, "Invalid input type to convertArgToPython");
  }
}

template <typename... Args>
static inline LazyKernelCompileResult runTritonKernelWithAutotune(
    PyObject* pending_kernels,
    const std::string& kernel_name,
    void* stream,
    const Args&... kernel_args) {
  py::gil_scoped_acquire_simple acquire;

  constexpr size_t num_args = sizeof...(Args);
  RAIIPyObject py_args_list = PyList_New(num_args);
  AOTI_TORCH_CHECK(py_args_list, "Failed to create args list");

  size_t idx = 0;
  auto add_arg = [&py_args_list, &idx](PyObject* py_arg) {
    AOTI_TORCH_CHECK(py_arg, "Failed to convert argument");
    PyList_SetItem(py_args_list, idx++, py_arg);
  };
  // Use array pack-expansion instead of a fold expression to avoid
  // hitting the compiler's expression-nesting limit when there are
  // hundreds of kernel arguments (e.g. combo kernels).
  int dummy[] = {0, (add_arg(convertArgToPython(kernel_args)), 0)...};
  (void)dummy;

  RAIIPyObject call_args = PyTuple_Pack(
      4,
      pending_kernels,
      PyUnicode_FromString(kernel_name.c_str()),
      PyLong_FromVoidPtr(stream),
      py_args_list.get());
  AOTI_TORCH_CHECK(call_args, "Failed to create call args");

  RAIIPyObject result =
      PyObject_CallObject(run_triton_kernel_with_autotune, call_args);
  AOTI_TORCH_CHECK(result, "Failed to run kernel with autotuning");

  return extractCompileResult(result);
}

static inline void startKernelCompile(
    PyObject* pending_kernels,
    const std::string& kernel_name,
    const std::string& kernel_source) {
  py::gil_scoped_acquire_simple acquire;

  RAIIPyObject py_name = PyUnicode_FromString(kernel_name.c_str());
  RAIIPyObject py_source = PyUnicode_FromString(kernel_source.c_str());
  AOTI_TORCH_CHECK(py_name && py_source, "Failed to create Python args");

  RAIIPyObject call_args =
      PyTuple_Pack(3, pending_kernels, py_name.get(), py_source.get());
  AOTI_TORCH_CHECK(call_args, "Failed to create call args");

  RAIIPyObject result = PyObject_CallObject(start_kernel_compile, call_args);
  AOTI_TORCH_CHECK(result, "Failed to start kernel compilation");
}
