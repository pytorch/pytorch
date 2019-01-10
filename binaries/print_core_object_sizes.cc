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

#include <iostream>

#include "caffe2/core/init.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/context.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/proto/caffe2.pb.h"

#define PRINT_SIZE(cls) \
  std::cout << "Size of " #cls ": " << sizeof(cls) << " bytes." \
            << std::endl;

int main(int /* unused */, char** /* unused */) {
  PRINT_SIZE(caffe2::Blob);
  PRINT_SIZE(caffe2::Tensor<caffe2::CPUContext>);
  PRINT_SIZE(caffe2::Tensor<caffe2::CUDAContext>);
  PRINT_SIZE(caffe2::CPUContext);
  PRINT_SIZE(caffe2::CUDAContext);
  PRINT_SIZE(caffe2::OperatorBase);
  PRINT_SIZE(caffe2::OperatorDef);
  PRINT_SIZE(caffe2::Operator<caffe2::CPUContext>);
  PRINT_SIZE(caffe2::Operator<caffe2::CUDAContext>);
  PRINT_SIZE(caffe2::TypeMeta);
  PRINT_SIZE(caffe2::Workspace);
  return 0;
}
