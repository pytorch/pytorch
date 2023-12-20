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
  ONEDNN = 3,
  X86 = 4,
};

constexpr auto kNoQEngine = QEngine::NoQEngine;
constexpr auto kFBGEMM = QEngine::FBGEMM;
constexpr auto kQNNPACK = QEngine::QNNPACK;
constexpr auto kONEDNN = QEngine::ONEDNN;
constexpr auto kX86 = QEngine::X86;

inline std::string toString(QEngine qengine) {
  switch (qengine) {
    case kNoQEngine:
      return "NoQEngine";
    case kFBGEMM:
      return "FBGEMM";
    case kQNNPACK:
      return "QNNPACK";
    case kONEDNN:
      return "ONEDNN";
    case kX86:
      return "X86";
    default:
      TORCH_CHECK(
          false, "Unrecognized Quantized Engine: ", static_cast<int>(qengine));
  }
}

} // namespace c10
