#pragma once

#include <torch/csrc/distributed/c10d/Work.hpp>

namespace c10d {

C10_EXPORT void register_work(
    const at::Tensor& tensor,
    const c10::intrusive_ptr<c10d::Work>& work);

} // namespace c10d
