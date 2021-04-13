#include "lazy_tensor_core/csrc/view.h"

#include <algorithm>
#include <functional>
#include <numeric>

#include "lazy_tensor_core/csrc/ops/as_strided.h"
#include "lazy_tensor_core/csrc/ops/as_strided_view_update.h"
#include "lazy_tensor_core/csrc/ops/diagonal.h"
#include "lazy_tensor_core/csrc/ops/diagonal_view_update.h"
#include "lazy_tensor_core/csrc/ops/generic_slice.h"
#include "lazy_tensor_core/csrc/ops/ops.h"
#include "lazy_tensor_core/csrc/ops/permute.h"
#include "lazy_tensor_core/csrc/ops/resize.h"
#include "lazy_tensor_core/csrc/ops/select.h"
#include "lazy_tensor_core/csrc/ops/unselect.h"
#include "lazy_tensor_core/csrc/ops/update_slice.h"
#include "lazy_tensor_core/csrc/ops/view.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/permutation_util.h"
#include "lazy_tensors/shape_util.h"
#include "lazy_tensors/util.h"

namespace torch_lazy_tensors {
namespace {

ir::Value ApplyViewInfo(ir::Value ir_value, const ViewInfo& view_info) {
  switch (view_info.view_type) {
    case ViewInfo::Type::kSelect:
      return ir::MakeNode<ir::ops::Select>(
          ir_value, view_info.select->dim, view_info.select->start,
          view_info.select->end, view_info.select->stride);
    case ViewInfo::Type::kNarrow:
      return ir::MakeNode<ir::ops::GenericSlice>(ir_value, view_info.indices,
                                                 view_info.shape.dimensions());
    case ViewInfo::Type::kNoOp:
      return ir_value;
    case ViewInfo::Type::kPermute:
      return ir::MakeNode<ir::ops::Permute>(ir_value, view_info.permutation);
    case ViewInfo::Type::kReshape:
      return ir::MakeNode<ir::ops::View>(
          ir_value, lazy_tensors::util::ToVector<lazy_tensors::int64>(
                        view_info.shape.dimensions()));
    case ViewInfo::Type::kResize:
      return ir::MakeNode<ir::ops::Resize>(
          ir_value, lazy_tensors::util::ToVector<lazy_tensors::int64>(
                        view_info.shape.dimensions()));
    case ViewInfo::Type::kAsStrided:
      return ir::MakeNode<ir::ops::AsStrided>(
          ir_value,
          lazy_tensors::util::ToVector<lazy_tensors::int64>(
              view_info.shape.dimensions()),
          view_info.as_strided->stride, view_info.as_strided->offset);
    case ViewInfo::Type::kDiagonal:
      return ir::MakeNode<ir::ops::Diagonal>(
          ir_value, view_info.diagonal->offset, view_info.diagonal->dim1,
          view_info.diagonal->dim2);
    default:
      LTC_ERROR() << "Invalid view type: "
                  << lazy_tensors::util::GetEnumValue(view_info.view_type);
  }
}

ir::Value ApplyUpdate(ir::Value ir_value,
                      const Alias::UpdateData& update_data) {
  // We first bring the source IR value forward, by reshaping and slicing.
  std::vector<ir::Value> tmp_values({ir_value});
  for (size_t i = 0; i < update_data.view_infos.size(); ++i) {
    const ViewInfo& view_info = update_data.view_infos[i];
    tmp_values.push_back(ApplyViewInfo(tmp_values.back(), view_info));
  }
  // We then move backward given the source update value, by reshaping and
  // slice-updating.
  ir::Value result = update_data.ir_value;
  for (size_t i = update_data.view_infos.size(); i > 0; --i) {
    const ViewInfo& view_info = update_data.view_infos[i - 1];
    switch (view_info.view_type) {
      case ViewInfo::Type::kSelect:
        result = ir::MakeNode<ir::ops::Unselect>(
            tmp_values[i - 1], result, view_info.select->dim,
            view_info.select->start, view_info.select->end,
            view_info.select->stride);
        break;
      case ViewInfo::Type::kNarrow:
        result = ir::MakeNode<ir::ops::UpdateSlice>(tmp_values[i - 1], result,
                                                    view_info.indices);
        break;
      case ViewInfo::Type::kNoOp:
        break;
      case ViewInfo::Type::kPermute:
        result = ir::MakeNode<ir::ops::Permute>(
            result, lazy_tensors::InversePermutation(view_info.permutation));
        break;
      case ViewInfo::Type::kReshape:
        result = ir::MakeNode<ir::ops::View>(
            result, lazy_tensors::util::ToVector<lazy_tensors::int64>(
                        view_info.source_shape.dimensions()));
        break;
      case ViewInfo::Type::kResize:
        result = ir::MakeNode<ir::ops::Resize>(
            result, lazy_tensors::util::ToVector<lazy_tensors::int64>(
                        view_info.source_shape.dimensions()));
        break;
      case ViewInfo::Type::kAsStrided:
        result = ir::MakeNode<ir::ops::AsStridedViewUpdate>(
            tmp_values[i - 1], result,
            lazy_tensors::util::ToVector<lazy_tensors::int64>(
                view_info.source_shape.dimensions()),
            view_info.as_strided->stride, view_info.as_strided->offset);
        break;
      case ViewInfo::Type::kDiagonal:
        result = ir::MakeNode<ir::ops::DiagonalViewUpdate>(
            tmp_values[i - 1], result, view_info.diagonal->offset,
            view_info.diagonal->dim1, view_info.diagonal->dim2);
        break;
      default:
        LTC_ERROR() << "Invalid view type: "
                    << lazy_tensors::util::GetEnumValue(view_info.view_type);
    }
  }
  return result;
}

}  // namespace

ViewInfo::ViewInfo(Type view_type, lazy_tensors::Shape shape,
                   lazy_tensors::Shape source_shape)
    : view_type(view_type),
      shape(std::move(shape)),
      indices(source_shape.rank(), 0),
      source_shape(std::move(source_shape)) {}

ViewInfo::ViewInfo(Type view_type, lazy_tensors::Shape source_shape,
                   std::vector<lazy_tensors::int64> permutation)
    : view_type(view_type),
      shape(ir::ops::Permute::MakePermuteShape(source_shape, permutation)),
      source_shape(std::move(source_shape)),
      permutation(std::move(permutation)) {
  LTC_CHECK(view_type == Type::kPermute);
}

ViewInfo::ViewInfo(Type view_type, const lazy_tensors::Shape& source_shape,
                   SelectInfo select)
    : view_type(view_type),
      shape(ir::ops::Select::MakeSelectShape(
          source_shape, select.dim, select.start, select.end, select.stride)),
      source_shape(source_shape),
      select(std::move(select)) {
  LTC_CHECK(view_type == Type::kSelect);
}

ViewInfo::ViewInfo(Type view_type, lazy_tensors::Shape shape,
                   lazy_tensors::Shape source_shape, AsStridedInfo as_strided)
    : view_type(view_type),
      shape(std::move(shape)),
      source_shape(std::move(source_shape)),
      as_strided(std::move(as_strided)) {
  LTC_CHECK(view_type == Type::kAsStrided);
}

ViewInfo::ViewInfo(Type view_type, const lazy_tensors::Shape& source_shape,
                   DiagonalInfo diagonal)
    : view_type(view_type),
      shape(ir::ops::Diagonal::MakeDiagonalShape(source_shape, diagonal.offset,
                                                 diagonal.dim1, diagonal.dim2)),
      source_shape(source_shape),
      diagonal(std::move(diagonal)) {
  LTC_CHECK(view_type == Type::kDiagonal);
}

void Alias::Update(ir::Value ir_value, std::vector<ViewInfo> view_infos) {
  if (!updates_.empty() && updates_.back().view_infos == view_infos) {
    updates_.back().ir_value = std::move(ir_value);
  } else {
    updates_.push_back({std::move(ir_value), std::move(view_infos)});
  }
  ++generation_;
}

ir::Value Alias::SyncUpdateOperations() {
  for (auto& update_data : updates_) {
    ir_value_ = ApplyUpdate(ir_value_, update_data);
  }
  updates_.clear();
  return ir_value_;
}

View::View(lazy_tensors::Shape shape, std::shared_ptr<Alias> alias,
           ViewInfo view_info)
    : shape_(std::move(shape)), alias_(std::move(alias)) {
  view_infos_.push_back(std::move(view_info));
}

View::View(lazy_tensors::Shape shape, std::shared_ptr<Alias> alias,
           std::vector<ViewInfo> view_infos)
    : view_infos_(std::move(view_infos)),
      shape_(std::move(shape)),
      alias_(std::move(alias)) {}

void View::Update(ir::Value ir_value) {
  alias_->Update(std::move(ir_value), view_infos_);
}

std::shared_ptr<View> View::CreateSubView(lazy_tensors::Shape shape,
                                          ViewInfo view_info) {
  std::vector<ViewInfo> view_infos(view_infos_);
  view_infos.push_back(std::move(view_info));
  return std::make_shared<View>(std::move(shape), alias_,
                                std::move(view_infos));
}

View::IrNode View::GetViewIrNode() {
  if (IsUpToDate()) {
    return {ir_value_, false};
  }
  ir::Value update = alias_->SyncUpdateOperations();
  for (auto& view_info : view_infos_) {
    update = ApplyViewInfo(update, view_info);
  }
  ir_value_ = update;
  generation_ = alias_->generation();
  return {ir_value_, true};
}

}  // namespace torch_lazy_tensors
