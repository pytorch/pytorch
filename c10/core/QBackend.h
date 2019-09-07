#pragma once

#include <c10/core/DeviceType.h>
#include <c10/core/TensorTypeId.h>
#include <c10/util/Exception.h>

namespace c10 {

/**
 * QBackend is an enum that is used to select the backend to run quantized ops.
 */
enum class QBackend : uint8_t {
  FBGEMM = 0,
  QNNPACK = 1,
  COMPILE_TIME_NUM_QBACKENDS = 2,
};

constexpr auto kFBGEMM = QBackend::FBGEMM;
constexpr auto kQNNPACK = QBackend::QNNPACK;
constexpr int COMPILE_TIME_NUM_QBACKENDS =
    static_cast<int>(QBackend::COMPILE_TIME_NUM_QBACKENDS);

inline std::string toString(QBackend qbackend) {
  switch (qbackend) {
    case kFBGEMM:
      return "FBGEMM";
    case kQNNPACK:
      return "QNNPACK";
    default:
      TORCH_CHECK(
          false,
          "Unrecognized Quantized Backend: ",
          static_cast<int>(qbackend));
  }
}

} // namespace c10
