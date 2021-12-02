#include <torch/csrc/lazy/core/view_ops/ltc_ops.h>

namespace torch {
namespace lazy {

const OpKindWrapper ltc_as_strided_view_update(
    "lazy_tensors::as_strided_view_update");
const OpKindWrapper ltc_diagonal_view_update(
    "lazy_tensors::diagonal_view_update");
const OpKindWrapper ltc_narrow_view_update("lazy_tensors::narrow_view_update");
const OpKindWrapper ltc_select_view_update("lazy_tensors::select_view_update");

} // namespace lazy
} // namespace torch
