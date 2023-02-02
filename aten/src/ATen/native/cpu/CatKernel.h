#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/core/IListRef.h>

namespace at { namespace native {

using cat_contig_fn = void(*)(const Tensor &, const MaterializedITensorListRef&, int64_t, bool);
DECLARE_DISPATCH(cat_contig_fn, cat_contig_stub);

}}  // namespace at::native
