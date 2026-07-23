/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ATen/Parallel.h"

#include <iostream>
#include <sstream>

#ifdef __linux__
#include <sys/types.h>
#include <unistd.h>
#endif

int main(int argc, char** argv) {
  at::init_num_threads();

  std::cout << at::get_parallel_info() << std::endl;

# ifdef __linux__
  std::ostringstream cmd;
  cmd << "lsof -p " << getpid() << " | grep .so";
  std::cout << "Loaded .so:" << std::endl;
  std::cout << cmd.str() << std::endl;
  std::system(cmd.str().c_str());
# endif

  return 0;
}
