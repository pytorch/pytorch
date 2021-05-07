#include "lazy_tensors/shape.h"

namespace lazy_tensors {

void Shape::DeleteDimension(int64 dim_to_delete) {
  LTC_CHECK(IsArray());
  LTC_CHECK_GE(dim_to_delete, 0);
  LTC_CHECK_LT(dim_to_delete, dimensions_.size());
  dimensions_.erase(dimensions_.begin() + dim_to_delete);
  for (int64 i = 0; i < layout_.minor_to_major().size();) {
    if (layout_.minor_to_major(i) == dim_to_delete) {
      layout_.mutable_minor_to_major()->erase(
          layout_.mutable_minor_to_major()->begin() + i);
      continue;
    }
    if (layout_.minor_to_major(i) > dim_to_delete) {
      (*layout_.mutable_minor_to_major())[i] -= 1;
    }
    ++i;
  }
}

bool Shape::IsDynamicMode() { return dynamic_mode_.load(); }

void Shape::SetDynamicMode() { dynamic_mode_ = true; }

std::atomic<bool> Shape::dynamic_mode_{false};

}  // namespace lazy_tensors
