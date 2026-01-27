#include <ATen/xpu/level_zero_stub/ATenLevelZero.h>

namespace at::xpu {

LevelZero* load_level_zero() {
  auto self = new LevelZero();
#define CREATE_ASSIGN(name) self->name = name;
  AT_FORALL_L0(CREATE_ASSIGN)
  return self;
}

} // namespace at::xpu
