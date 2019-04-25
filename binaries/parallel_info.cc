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

// This script converts the MNIST dataset to leveldb.
// The MNIST dataset could be downloaded at
//    http://yann.lecun.com/exdb/mnist/


#include "ATen/Parallel.h"

#include <iostream>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef TH_BLAS_MKL
#include <mkl.h>
#endif

#ifdef __linux__
#include <sys/types.h>
#include <unistd.h>
#endif

const char* get_env_var(const char* var_name) {
  const char* value = std::getenv(var_name);
  return value ? value : "[not set]";
}

int main(int argc, char** argv) {
  at::init_num_threads();

  std::cout << "std::thread::hardware_concurrency() :\t"
            << std::thread::hardware_concurrency() << std::endl;
  std::cout << std::endl;

  std::cout << "ATen/Parallel:" << std::endl;
  std::cout << "\tat::get_num_threads()\t:\t"
            << at::get_num_threads() << std::endl;
  std::cout << std::endl;

  std::cout << "OpenMP:" << std::endl;
# ifdef _OPENMP
  std::cout << "\tomp_get_max_threads()\t:\t"
            << omp_get_max_threads() << std::endl;
# else
  std::cout << "\tnot available" << std::endl;
# endif
  std::cout << std::endl;

  std::cout << "MKL:" << std::endl;
# ifdef TH_BLAS_MKL
  std::cout << "\tmkl_get_max_threads()\t:\t"
            << mkl_get_max_threads() << std::endl;
# else
  std::cout << "\tnot available" << std::endl;
# endif
  std::cout << std::endl;

  std::cout << "Environment variables:" << std::endl;
  std::cout << "\tOMP_NUM_THREADS\t:\t"
            << get_env_var("OMP_NUM_THREADS") << std::endl;
  std::cout << "\tMKL_NUM_THREADS\t:\t"
            << get_env_var("MKL_NUM_THREADS") << std::endl;
  std::cout << std::endl;

# ifdef __linux__
  std::ostringstream cmd;
  cmd << "lsof -p " << getpid() << " | grep .so";
  std::cout << "Loaded .so:" << std::endl;
  std::cout << cmd.str() << std::endl;
  std::system(cmd.str().c_str());
# endif

  return 0;
}
