// The clang-tidy job seems to complain that it can't find cudnn.h without this.
// This file should only be compiled if this condition holds, so it should be
// safe.
#if defined(USE_CUDNN) || defined(USE_ROCM)
#include <torch/csrc/utils/pybind.h>

#include <tuple>

namespace {
using version_tuple = std::tuple<size_t, size_t, size_t>;
}

#ifdef USE_CUDNN
#include <cudnn.h>

namespace {

version_tuple getCompileVersion() {
  return version_tuple(CUDNN_MAJOR, CUDNN_MINOR, CUDNN_PATCHLEVEL);
}

version_tuple getRuntimeVersion() {
#ifndef USE_STATIC_CUDNN
  int major = 0, minor = 0, patch = 0;
  cudnnGetProperty(MAJOR_VERSION, &major);
  cudnnGetProperty(MINOR_VERSION, &minor);
  cudnnGetProperty(PATCH_LEVEL, &patch);
  return version_tuple((size_t)major, (size_t)minor, (size_t)patch);
#else
  return getCompileVersion();
#endif
}

size_t getVersionInt() {
#ifndef USE_STATIC_CUDNN
  return cudnnGetVersion();
#else
  return CUDNN_VERSION;
#endif
}

} // namespace
#elif defined(USE_ROCM)
#include <miopen/miopen.h>
#include <miopen/version.h>

namespace {

version_tuple getCompileVersion() {
  return version_tuple(
      MIOPEN_VERSION_MAJOR, MIOPEN_VERSION_MINOR, MIOPEN_VERSION_PATCH);
}

version_tuple getRuntimeVersion() {
  // MIOpen doesn't include runtime version info before 2.3.0
#if (MIOPEN_VERSION_MAJOR > 2) || \
    (MIOPEN_VERSION_MAJOR == 2 && MIOPEN_VERSION_MINOR > 2)
  size_t major, minor, patch;
  miopenGetVersion(&major, &minor, &patch);
  return version_tuple(major, minor, patch);
#else
  return getCompileVersion();
#endif
}

size_t getVersionInt() {
  // miopen version is MAJOR*1000000 + MINOR*1000 + PATCH
  auto [major, minor, patch] = getRuntimeVersion();
  return major * 1000000 + minor * 1000 + patch;
}

} // namespace
#endif

namespace torch::cuda::shared {

void initCudnnBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto cudnn = m.def_submodule("_cudnn", "libcudnn.so bindings");

  py::enum_<cudnnRNNMode_t>(cudnn, "RNNMode")
      .value("rnn_relu", CUDNN_RNN_RELU)
      .value("rnn_tanh", CUDNN_RNN_TANH)
      .value("lstm", CUDNN_LSTM)
      .value("gru", CUDNN_GRU);

  // The runtime version check in python needs to distinguish cudnn from miopen
#ifdef USE_CUDNN
  cudnn.attr("is_cuda") = true;
#else
  cudnn.attr("is_cuda") = false;
#endif

  cudnn.def("getRuntimeVersion", getRuntimeVersion);
  cudnn.def("getCompileVersion", getCompileVersion);
  cudnn.def("getVersionInt", getVersionInt);
}

} // namespace torch::cuda::shared
#endif
