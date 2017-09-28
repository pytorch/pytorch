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

#include "redis_store_handler.h"

#include <caffe2/core/operator.h>

#include <string>

namespace caffe2 {

template <class Context>
class RedisStoreHandlerCreateOp final : public Operator<Context> {
 public:
  explicit RedisStoreHandlerCreateOp(
      const OperatorDef& operator_def,
      Workspace* ws)
      : Operator<Context>(operator_def, ws),
        host_(
            OperatorBase::template GetSingleArgument<std::string>("host", "")),
        port_(OperatorBase::template GetSingleArgument<int>("port", 0)),
        prefix_(OperatorBase::template GetSingleArgument<std::string>(
            "prefix",
            "")) {
    CAFFE_ENFORCE_NE(host_, "", "host is a required argument");
    CAFFE_ENFORCE_NE(port_, 0, "port is a required argument");
  }

  bool RunOnDevice() override {
    auto ptr = std::unique_ptr<StoreHandler>(
        new RedisStoreHandler(host_, port_, prefix_));
    *OperatorBase::Output<std::unique_ptr<StoreHandler>>(HANDLER) =
        std::move(ptr);
    return true;
  }

 private:
  std::string host_;
  int port_;
  std::string prefix_;

  OUTPUT_TAGS(HANDLER);
};

} // namespace caffe2
