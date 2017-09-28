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

#include "file_store_handler.h"

#include <caffe2/core/operator.h>

namespace caffe2 {

template <class Context>
class FileStoreHandlerCreateOp final : public Operator<Context> {
 public:
  explicit FileStoreHandlerCreateOp(
      const OperatorDef& operator_def,
      Workspace* ws)
      : Operator<Context>(operator_def, ws),
        basePath_(
            OperatorBase::template GetSingleArgument<std::string>("path", "")),
        prefix_(OperatorBase::template GetSingleArgument<std::string>(
            "prefix",
            "")) {
    CAFFE_ENFORCE_NE(basePath_, "", "path is a required argument");
  }

  bool RunOnDevice() override {
    auto ptr =
        std::unique_ptr<StoreHandler>(new FileStoreHandler(basePath_, prefix_));
    *OperatorBase::Output<std::unique_ptr<StoreHandler>>(HANDLER) =
        std::move(ptr);
    return true;
  }

 private:
  std::string basePath_;
  std::string prefix_;

  OUTPUT_TAGS(HANDLER);
};

} // namespace caffe2
