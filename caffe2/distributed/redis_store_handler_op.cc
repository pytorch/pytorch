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

#include "redis_store_handler_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    RedisStoreHandlerCreate,
    RedisStoreHandlerCreateOp<CPUContext>);

OPERATOR_SCHEMA(RedisStoreHandlerCreate)
    .NumInputs(0)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Creates a unique_ptr<StoreHandler> that uses a Redis server as backing store.
)DOC")
    .Arg("host", "host name of Redis server")
    .Arg("port", "port number of Redis server")
    .Arg("prefix", "keys used by this instance are prefixed with this string")
    .Output(0, "handler", "unique_ptr<StoreHandler>");

NO_GRADIENT(RedisStoreHandlerCreateOp);

} // namespace caffe2
