#pragma once

#include <exception>

#include "caffe2/core/blob.h"

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
