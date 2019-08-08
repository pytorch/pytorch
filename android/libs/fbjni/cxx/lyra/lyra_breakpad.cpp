/*
 *  Copyright (c) 2004-present, Facebook, Inc.
 *
 *  This source code is licensed under the MIT license found in the LICENSE
 *  file in the root directory of this source tree.
 *
 */
#include <lyra/lyra.h>

namespace facebook {
namespace lyra {

/**
 * This can be overridden by an implementation capable of looking up
 * the breakpad id for logging purposes.
*/
__attribute__((weak))
std::string getBreakpadId(const std::string& library) {
  return "<unimplemented>";
}

}
}
