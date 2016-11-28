#pragma once

#include <cstdint>

namespace thd {

enum Functions: std::uint16_t {
  construct,
  constructWithSize,
  free,
  resize,
  resizeAs,
  resize1d,
  resize2d,
  resize3d,
  resize4d,
  resize5d,
  set,
  setStorage,
  setStorage1d,
  setStorage2d,
  setStorage3d,
  setStorage4d,
  narrow,
  select,
  add,
  fill,

  // storage functions
  storageConstruct,
  storageConstructWithSize,
  storageFree,
  storageResize,
  storageSet,

  // communication requests
  sendTensor,
  sendStorage,
};

} // namespace thd
