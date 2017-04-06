/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/common/linux.h"

#include <fstream>
#include <mutex>

#include "gloo/common/logging.h"

namespace gloo {

const std::set<std::string>& kernelModules() {
  static std::once_flag once;
  static std::set<std::string> modules;

  std::call_once(once, [](){
      std::ifstream ifs("/proc/modules");
      int n = 0;

      std::string line;
      while (std::getline(ifs, line)) {
        auto sep = line.find(' ');
        GLOO_ENFORCE_NE(sep, std::string::npos);
        modules.insert(line.substr(0, sep));
      }
    });

  return modules;
}

} // namespace gloo
