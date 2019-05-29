#pragma once

#include <cstdint>

namespace c10 {

enum class AliasAnalysisKind : uint8_t {
  DEFAULT, // The most conservative alias analysis type, assumes side-effects
  PURE
};

}
