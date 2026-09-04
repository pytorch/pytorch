#pragma once

#include <c10/core/TensorImpl.h>
#include <torch/csrc/jit/serialization/pickler.h>

namespace c10::openreg {
struct OpenRegBackendMeta : public c10::BackendMeta {
  OpenRegBackendMeta(int version_number, int format_number)
      : version_number_(version_number), format_number_(format_number) {}

  int version_number_{-1};
  int format_number_{-1};
};
} // namespace c10::openreg

#define REGISTER_PRIVATEUSE1_SERIALIZATION(                                    \
    FOR_SERIALIZATION, FOR_DESERIALIZATION)                                    \
  static int register_serialization() {                                        \
    torch::jit::TensorBackendMetaRegistry(                                     \
        c10::DeviceType::PrivateUse1, FOR_SERIALIZATION, FOR_DESERIALIZATION); \
    return 0;                                                                  \
  }                                                                            \
  static const int _temp = register_serialization();
