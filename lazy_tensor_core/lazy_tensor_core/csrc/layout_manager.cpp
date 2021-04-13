#include "lazy_tensor_core/csrc/layout_manager.h"

#include <algorithm>
#include <exception>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>

#include "absl/strings/str_split.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/ltc_logging.h"
#include "lazy_tensors/computation_client/sys_util.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/shape_util.h"

namespace torch_lazy_tensors {
namespace {

class LayoutManager {
 public:
  static LayoutManager* Get() {
    static LayoutManager* mgr = new LayoutManager();
    return mgr;
  }

  const std::vector<lazy_tensors::int64>* GetLayout(
      lazy_tensors::Span<const lazy_tensors::int64> dimensions) const {
    auto it = layouts_.find(dimensions);
    return it != layouts_.end() ? &it->second->layout : nullptr;
  }

 private:
  struct LayoutEntry {
    std::vector<lazy_tensors::int64> dimensions;
    std::vector<lazy_tensors::int64> layout;
  };

  struct DimensionsHasher {
    size_t operator()(
        const lazy_tensors::Span<const lazy_tensors::int64>& dimensions) const {
      return lazy_tensors::util::HashReduce(
          lazy_tensors::util::MHash(dimensions));
    }
  };

  using LayoutMap =
      std::unordered_map<lazy_tensors::Span<const lazy_tensors::int64>,
                         std::shared_ptr<LayoutEntry>, DimensionsHasher>;

  LayoutManager() {
    try {
      PopulateLayouts();
    } catch (const std::exception& ex) {
      LTC_LOG(FATAL) << "Exception caught while parsing layouts: " << ex.what();
    }
  }

  void PopulateLayouts() {
    // Layouts: SHAPE=LAYOUT;...
    // SHAPE: INT,...
    // LAYOUT: INT,...
    std::string layouts_env =
        lazy_tensors::sys_util::GetEnvString("LTC_LAYOUTS", "");
    if (!layouts_env.empty()) {
      std::vector<std::string> layouts = absl::StrSplit(layouts_env, ';');
      for (const auto& layout_str : layouts) {
        std::vector<std::string> parts = absl::StrSplit(layout_str, '=');
        LTC_CHECK_EQ(parts.size(), 2) << layout_str;

        auto entry = std::make_shared<LayoutEntry>();
        entry->dimensions = ParseIntList(parts[0]);
        entry->layout = ParseLayout(parts[1], entry->dimensions.size());
        layouts_.emplace(entry->dimensions, entry);

        LTC_VLOG(2) << "Registering layout " << parts[1] << " for shape "
                    << parts[0];
      }
    }
  }

  static std::vector<lazy_tensors::int64> ParseIntList(
      const std::string& list_str) {
    std::vector<std::string> parts = absl::StrSplit(list_str, ',');
    std::vector<lazy_tensors::int64> ints;
    for (const auto& int_str : parts) {
      ints.push_back(std::stol(int_str));
    }
    return ints;
  }

  static std::vector<lazy_tensors::int64> ParseLayout(
      const std::string& list_str, lazy_tensors::int64 rank) {
    std::vector<lazy_tensors::int64> ints = ParseIntList(list_str);
    LTC_CHECK_EQ(ints.size(), rank) << list_str;
    std::set<lazy_tensors::int64> unique_ints;
    for (auto dim : ints) {
      LTC_CHECK_GE(dim, 0) << list_str;
      LTC_CHECK_LT(dim, rank) << list_str;
      unique_ints.insert(dim);
    }
    LTC_CHECK_EQ(unique_ints.size(), rank) << list_str;
    return ints;
  }

  LayoutMap layouts_;
};

double PaddingFactor(lazy_tensors::int64 size, int padding) {
  int rem = static_cast<int>(size % padding);
  return 1.0 + (rem > 0 ? static_cast<double>(padding - rem) /
                              static_cast<double>(size)
                        : 0.0);
}

lazy_tensors::Shape MakeShapeWithSortedLayout(
    lazy_tensors::Span<const lazy_tensors::int64> dimensions,
    lazy_tensors::PrimitiveType type) {
  // Place bigger dimensions on most minor layout locations.
  std::vector<lazy_tensors::int64> layout =
      lazy_tensors::util::Iota<lazy_tensors::int64>(dimensions.size(),
                                                    dimensions.size() - 1, -1);
  std::sort(layout.begin(), layout.end(),
            [&](lazy_tensors::int64 a, lazy_tensors::int64 b) {
              return dimensions[a] > dimensions[b];
            });
  return lazy_tensors::ShapeUtil::MakeShapeWithLayout(type, dimensions, layout);
}

lazy_tensors::Shape* SetDynamicDimensions(
    lazy_tensors::Shape* shape,
    lazy_tensors::Span<const bool> dynamic_dimensions) {
  if (!dynamic_dimensions.empty()) {
    LTC_CHECK_EQ(dynamic_dimensions.size(), shape->rank());
    for (size_t i = 0; i < dynamic_dimensions.size(); ++i) {
      shape->set_dynamic_dimension(i, dynamic_dimensions[i]);
    }
  }
  return shape;
}

lazy_tensors::Shape MakeTpuShape(
    lazy_tensors::Span<const lazy_tensors::int64> dimensions,
    lazy_tensors::Span<const bool> dynamic_dimensions,
    lazy_tensors::PrimitiveType type) {
  static double max_padding_factor =
      lazy_tensors::sys_util::GetEnvDouble("LTC_MAX_PADDING_FACTOR", 1.25);
  lazy_tensors::Shape shape;
  if (PaddingFactor(dimensions[dimensions.size() - 1], 128) *
          PaddingFactor(dimensions[dimensions.size() - 2], 8) <
      max_padding_factor) {
    shape = lazy_tensors::ShapeUtil::MakeShapeWithDescendingLayout(type,
                                                                   dimensions);
  } else {
    shape = MakeShapeWithSortedLayout(dimensions, type);
  }
  SetDynamicDimensions(&shape, dynamic_dimensions);
  return shape;
}

lazy_tensors::Shape MakeShapeWithLayout(
    lazy_tensors::PrimitiveType type,
    lazy_tensors::Span<const lazy_tensors::int64> dimensions,
    lazy_tensors::Span<const bool> dynamic_dimensions,
    lazy_tensors::Span<const lazy_tensors::int64> layout) {
  lazy_tensors::Shape shape =
      lazy_tensors::ShapeUtil::MakeShapeWithLayout(type, dimensions, layout);
  SetDynamicDimensions(&shape, dynamic_dimensions);
  return shape;
}

}  // namespace

lazy_tensors::Shape MakeTorchTensorLayout(
    lazy_tensors::Span<const lazy_tensors::int64> dimensions,
    lazy_tensors::Span<const bool> dynamic_dimensions,
    lazy_tensors::PrimitiveType type) {
  lazy_tensors::Shape shape =
      lazy_tensors::ShapeUtil::MakeShapeWithDescendingLayout(type, dimensions);
  SetDynamicDimensions(&shape, dynamic_dimensions);
  return shape;
}

lazy_tensors::Shape MakeArrayShapeFromDimensions(
    lazy_tensors::Span<const lazy_tensors::int64> dimensions,
    lazy_tensors::Span<const bool> dynamic_dimensions,
    lazy_tensors::PrimitiveType type, DeviceType device_type) {
  auto layout_ptr = LayoutManager::Get()->GetLayout(dimensions);
  if (layout_ptr != nullptr) {
    return MakeShapeWithLayout(type, dimensions, dynamic_dimensions,
                               *layout_ptr);
  }
  if (dimensions.size() > 1 && device_type == DeviceType::TPU) {
    return MakeTpuShape(dimensions, dynamic_dimensions, type);
  }
  return MakeTorchTensorLayout(dimensions, dynamic_dimensions, type);
}

}  // namespace torch_lazy_tensors
