#pragma once

#include <cstdint>

namespace thd {

enum Functions: std::uint16_t {
  construct,
	constructWithSize,
  add,
  free
};

} // namespace thd
