#pragma once

#include <ATen/FunctionalStorageImpl.h>

namespace at::functionalization {

// `ViewMeta` implementation for `resize_` operation.
struct TORCH_API resize__ViewMeta : public ViewMeta {
  FUNCTIONALIZATION_VIEWMETA_NAME(resize__ViewMeta);
  FUNCTIONALIZATION_VIEWMETA_SERIALIZABLE_TUPLE(
      bool /* reapply_views */,
      const std::vector<int64_t>&);

  resize__ViewMeta(const SerializableTuple& tpl)
      : resize__ViewMeta(std::get<0>(tpl), std::get<1>(tpl)) {}

  resize__ViewMeta(bool reapply_views, const std::vector<int64_t>& size)
      : ViewMeta(/*has_symbolic_inputs=*/false),
        reapply_views(reapply_views),
        size(size) {}

  Tensor forward(const Tensor& base) override;
  Tensor reverse(const Tensor& base, const Tensor& mutated_view) override;

  SerializableTuple to_serializable_tuple() {
    return std::make_tuple(reapply_views, size);
  }

  bool reapply_views;
  std::vector<int64_t> size;
};

// `ViewMeta` implementation for `_unsafe_view` operation.
struct TORCH_API _unsafe_view_ViewMeta : public ViewMeta {
  FUNCTIONALIZATION_VIEWMETA_NAME(_unsafe_view_ViewMeta);
  FUNCTIONALIZATION_VIEWMETA_SERIALIZABLE_TUPLE(
      bool /* has_symbolic_inputs */,
      const std::vector<c10::SymInt>&);

  _unsafe_view_ViewMeta(const SerializableTuple& tpl)
      : _unsafe_view_ViewMeta(std::get<0>(tpl), std::get<1>(tpl)) {}

  _unsafe_view_ViewMeta(
      bool has_symbolic_inputs,
      const std::vector<c10::SymInt>& size)
      : ViewMeta(has_symbolic_inputs), size(size) {}

  Tensor forward(const Tensor& base) override;
  Tensor reverse(const Tensor& base, const Tensor& mutated_view) override;

  SerializableTuple to_serializable_tuple() {
    return std::make_tuple(has_symbolic_inputs, size);
  }

  std::vector<c10::SymInt> size;
};

} // namespace at::functionalization
