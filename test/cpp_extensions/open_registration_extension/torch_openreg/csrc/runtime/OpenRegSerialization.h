#include <torch/csrc/jit/serialization/pickler.h>

#define REGISTER_PRIVATEUSE1_SERIALIZATION(                                    \
    FOR_SERIALIZATION, FOR_DESERIALIZATION)                                    \
  static int register_serialization() {                                        \
    torch::jit::TensorBackendMetaRegistry(                                     \
        c10::DeviceType::PrivateUse1, FOR_SERIALIZATION, FOR_DESERIALIZATION); \
    return 0;                                                                  \
  }                                                                            \
  static const int _temp = register_serialization();
