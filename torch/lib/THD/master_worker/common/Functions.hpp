#pragma once

#include <cstdint>

namespace thd {

enum Functions: std::uint16_t {
  tensorConstruct,
  tensorConstructWithSize,
  tensorFree,
  tensorResize,
  tensorResizeAs,
  tensorResize1d,
  tensorResize2d,
  tensorResize3d,
  tensorResize4d,
  tensorResize5d,
  tensorSet,
  tensorSetStorage,
  tensorSetStorage1d,
  tensorSetStorage2d,
  tensorSetStorage3d,
  tensorSetStorage4d,
  tensorNarrow,
  tensorSelect,
  tensorTranspose,
  tensorUnfold,
  tensorAdd,
  tensorFill,

  // storage functions
  storageSet,
  storageGet,

  storageConstruct,
  storageConstructWithSize,
  storageConstructWithSize1,
  storageConstructWithSize2,
  storageConstructWithSize3,
  storageConstructWithSize4,

  storageFree,
  storageResize,
  storageFill,

  // communication requests
  sendTensor,
  sendStorage,
};

} // namespace thd
