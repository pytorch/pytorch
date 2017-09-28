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

#include "file_store_handler_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    FileStoreHandlerCreate,
    FileStoreHandlerCreateOp<CPUContext>);

OPERATOR_SCHEMA(FileStoreHandlerCreate)
    .NumInputs(0)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Creates a unique_ptr<StoreHandler> that uses the filesystem as backing
store (typically a filesystem shared between many nodes, such as NFS).
This store handler is not built to be fast. Its recommended use is for
integration tests and prototypes where extra dependencies are
cumbersome. Use an ephemeral path to ensure multiple processes or runs
don't interfere.
)DOC")
    .Arg("path", "base path used by the FileStoreHandler")
    .Arg("prefix", "prefix for all keys used by this store")
    .Output(0, "handler", "unique_ptr<StoreHandler>");

NO_GRADIENT(FileStoreHandlerCreateOp);

} // namespace caffe2
