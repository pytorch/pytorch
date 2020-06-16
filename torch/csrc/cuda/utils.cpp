#include <torch/csrc/python_headers.h>
#include <stdarg.h>
#include <string>
#include <torch/csrc/cuda/THCP.h>

#include <torch/csrc/cuda/override_macros.h>

#define THC_GENERIC_FILE "torch/csrc/generic/utils.cpp"
#include <THH/THHGenerateAllTypes.h>

#define THC_GENERIC_FILE "torch/csrc/generic/utils.cpp"
#include <THH/THHGenerateComplexTypes.h>

#define THC_GENERIC_FILE "torch/csrc/generic/utils.cpp"
#include <THH/THHGenerateBoolType.h>

#define THC_GENERIC_FILE "torch/csrc/generic/utils.cpp"
#include <THH/THHGenerateBFloat16Type.h>

#ifdef USE_ROCM
// NB: It's a list of *optional* HIPStreamMasqueradingAsCUDA; when nullopt, that means to use
// whatever the current stream of the device the input is associated with was.
std::vector<c10::optional<at::hip::HIPStreamMasqueradingAsCUDA>> THPUtils_PySequence_to_CUDAStreamList(PyObject *obj) {
  if (!PySequence_Check(obj)) {
    throw std::runtime_error("Expected a sequence in THPUtils_PySequence_to_CUDAStreamList");
  }
  THPObjectPtr seq = THPObjectPtr(PySequence_Fast(obj, nullptr));
  if (seq.get() == nullptr) {
    throw std::runtime_error("expected PySequence, but got " + std::string(THPUtils_typename(obj)));
  }

  std::vector<c10::optional<at::hip::HIPStreamMasqueradingAsCUDA>> streams;
  Py_ssize_t length = PySequence_Fast_GET_SIZE(seq.get());
  for (Py_ssize_t i = 0; i < length; i++) {
    PyObject *stream = PySequence_Fast_GET_ITEM(seq.get(), i);

    if (PyObject_IsInstance(stream, THCPStreamClass)) {
      // Spicy hot reinterpret cast!!
      streams.emplace_back( at::hip::HIPStreamMasqueradingAsCUDA::unpack((reinterpret_cast<THCPStream*>(stream))->cdata) );
    } else if (stream == Py_None) {
      streams.emplace_back();
    } else {
      std::runtime_error("Unknown data type found in stream list. Need torch.cuda.Stream or None");
    }
  }
  return streams;
}

#endif
