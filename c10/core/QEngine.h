#pragma once

#include <c10/core/DeviceType.h>
#include <c10/core/DispatchKey.h>
#include <c10/util/Exception.h>

namespace c10 {

/**
 * QEngine is an enum that is used to select the engine to run quantized ops.
 * Keep this enum in sync with get_qengine_id() in
 * torch/backends/quantized/__init__.py
 */
enum class QEngine : uint8_t {
  NoQEngine = 0,
  FBGEMM = 1,
  QNNPACK = 2,
};

constexpr auto kNoQEngine = QEngine::NoQEngine;
constexpr auto kFBGEMM = QEngine::FBGEMM;
constexpr auto kQNNPACK = QEngine::QNNPACK;

inline std::string toString(QEngine qengine) {
  switch (qengine) {
    case kNoQEngine:
      return "NoQEngine";
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
