#pragma once

#include "InitMethod.hpp"

namespace thd {

struct InitMethodEnv : InitMethod {
  InitMethodEnv();
  virtual ~InitMethodEnv();

  InitMethod::Config getConfig() override;
};

} // namespace thd
