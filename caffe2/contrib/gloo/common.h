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

#include <exception>

#include "caffe2/core/blob.h"

#include <gloo/config.h>
#include <gloo/transport/device.h>

namespace caffe2 {
namespace gloo {

void signalFailure(Blob* status_blob, std::exception& exception);

struct createDeviceAttr {
    // "tcp" or "ibverbs"
    std::string transport;

    // E.g. "eth0" (tcp), or "mlx5_0" (ibverbs).
    // This may be empty to make Gloo figure it out.
    std::string interface;
};

std::shared_ptr<::gloo::transport::Device> createDevice(
    const createDeviceAttr attr);

} // namespace gloo
} // namespace caffe2
