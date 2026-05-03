#pragma once

namespace at::native::mps {

void scan_simple_mps_impl(
    const Tensor& self,
    const Tensor& output,
    int64_t dim,
    const std::string& op_name);

} // namespace at::native::mps
