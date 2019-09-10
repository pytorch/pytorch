#pragma once

#include <c10/core/DeviceType.h>
#include <c10/core/TensorTypeId.h>
#include <c10/util/Exception.h>

namespace c10 {

/**
 * QEngine is an enum that is used to select the engine to run quantized ops.
 */
enum class QEngine : uint8_t {
  FBGEMM = 0,
  QNNPACK = 1,
  COMPILE_TIME_NUM_QENGINES = 2,
};

constexpr auto kFBGEMM = QEngine::FBGEMM;
constexpr auto kQNNPACK = QEngine::QNNPACK;
constexpr int COMPILE_TIME_NUM_QENGINES =
    static_cast<int>(QEngine::COMPILE_TIME_NUM_QENGINES);

inline std::string toString(QEngine qengine) {
  switch (qengine) {
    case kFBGEMM:
      return "FBGEMM";
    case kQNNPACK:
      return "QNNPACK";
    default:
      TORCH_CHECK(
          false,
          "Unrecognized Quantized Engine: ",
          static_cast<int>(qengine));
  }
}

} // namespace c10
