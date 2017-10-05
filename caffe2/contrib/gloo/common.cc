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

#include "caffe2/contrib/gloo/common.h"

#include "caffe2/core/logging.h"
#include "caffe2/core/tensor.h"

#include <gloo/transport/tcp/device.h>
#if defined(GLOO_USE_IBVERBS) && GLOO_USE_IBVERBS
#include <gloo/transport/ibverbs/device.h>
#endif

namespace caffe2 {
namespace gloo {

void signalFailure(Blob* status_blob, std::exception& /* unused */) {
  auto* res = status_blob->GetMutable<TensorCPU>();
  res->Resize(1);
  res->template mutable_data<int32_t>()[0] = 1;
}

std::shared_ptr<::gloo::transport::Device> createDevice(
    const createDeviceAttr attr) {
  if (attr.transport == "tcp") {
    ::gloo::transport::tcp::attr tcpAttr;
    if (attr.interface.size() > 0) {
      tcpAttr.iface = attr.interface;
    }
    return ::gloo::transport::tcp::CreateDevice(tcpAttr);
  } else if (attr.transport == "ibverbs") {
#if defined(GLOO_USE_IBVERBS) && GLOO_USE_IBVERBS
    ::gloo::transport::ibverbs::attr ibverbsAttr;
    ibverbsAttr.port = 1;
    ibverbsAttr.index = 0;
    if (attr.interface.size() > 0) {
      ibverbsAttr.name = attr.interface;
    }
    return ::gloo::transport::ibverbs::CreateDevice(ibverbsAttr);
#else
    CAFFE_THROW(
      "Gloo was not compiled with ibverbs support. ",
      "Please recompile with -DUSE_IBVERBS=1.");
#endif
  }

  CAFFE_THROW("Invalid transport: ", attr.transport);
}

} // namespace gloo
} // namespace caffe2
