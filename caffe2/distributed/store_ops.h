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

#pragma once

#include "store_handler.h"

#include <caffe2/core/operator.h>

namespace caffe2 {

class StoreSetOp final : public Operator<CPUContext> {
 public:
  StoreSetOp(const OperatorDef& operator_def, Workspace* ws);
  bool RunOnDevice() override;

 private:
  std::string blobName_;

  INPUT_TAGS(HANDLER, DATA);
};

class StoreGetOp final : public Operator<CPUContext> {
 public:
  StoreGetOp(const OperatorDef& operator_def, Workspace* ws);
  bool RunOnDevice() override;

 private:
  std::string blobName_;

  INPUT_TAGS(HANDLER);
  OUTPUT_TAGS(DATA);
};

class StoreAddOp final : public Operator<CPUContext> {
 public:
  StoreAddOp(const OperatorDef& operator_def, Workspace* ws);
  bool RunOnDevice() override;

 private:
  std::string blobName_;
  int addValue_;

  INPUT_TAGS(HANDLER);
  OUTPUT_TAGS(VALUE);
};

class StoreWaitOp final : public Operator<CPUContext> {
 public:
  StoreWaitOp(const OperatorDef& operator_def, Workspace* ws);
  bool RunOnDevice() override;

 private:
  std::vector<std::string> blobNames_;

  INPUT_TAGS(HANDLER);
};
}
