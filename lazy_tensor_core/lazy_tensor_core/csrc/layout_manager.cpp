#include "lazy_tensor_core/csrc/layout_manager.h"

#include <algorithm>
#include <exception>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>

#include "lazy_tensors/computation_client/sys_util.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/shape_util.h"
#include "lazy_tensors/str_split.h"

namespace torch_lazy_tensors {
namespace {

class LayoutManager {
 public:
  static LayoutManager* Get() {
    static LayoutManager* mgr = new LayoutManager();
    return mgr;
  }

  const std::vector<int64_t>* GetLayout(
      c10::ArrayRef<int64_t> dimensions) const {
    auto it = layouts_.find(dimensions);
    return it != layouts_.end() ? &it->second->layout : nullptr;
  }

 private:
  struct LayoutEntry {
    std::vector<int64_t> dimensions;
    std::vector<int64_t> layout;
  };

  struct DimensionsHasher {
    size_t operator()(const c10::ArrayRef<int64_t>& dimensions) const {
      return torch::lazy::HashReduce(
          torch::lazy::MHash(dimensions));
    }
  };

  using LayoutMap =
      std::unordered_map<c10::ArrayRef<int64_t>, std::shared_ptr<LayoutEntry>,
                         DimensionsHasher>;

  LayoutManager() {
    try {
      PopulateLayouts();
    } catch (const std::exception& ex) {
      LOG(FATAL) << "Exception caught while parsing layouts: " << ex.what();
    }
  }

  void PopulateLayouts() {
    // Layouts: SHAPE=LAYOUT;...
    // SHAPE: INT,...
    // LAYOUT: INT,...
    std::string layouts_env =
        lazy_tensors::sys_util::GetEnvString("LTC_LAYOUTS", "");
    if (!layouts_env.empty()) {
      std::vector<std::string> layouts =
          lazy_tensors::StrSplit(layouts_env, ';');
      for (const auto& layout_str : layouts) {
        std::vector<std::string> parts =
            lazy_tensors::StrSplit(layout_str, '=');
        CHECK_EQ(parts.size(), 2) << layout_str;

        auto entry = std::make_shared<LayoutEntry>();
        entry->dimensions = ParseIntList(parts[0]);
        entry->layout = ParseLayout(parts[1], entry->dimensions.size());
        layouts_.emplace(entry->dimensions, entry);

        VLOG(2) << "Registering layout " << parts[1] << " for shape "
                << parts[0];
      }
    }
  }

  static std::vector<int64_t> ParseIntList(const std::string& list_str) {
    std::vector<std::string> parts = lazy_tensors::StrSplit(list_str, ',');
    std::vector<int64_t> ints;
    for (const auto& int_str : parts) {
      ints.push_back(std::stol(int_str));
    }
    return ints;
  }
  // TODO(whc) see if this stuff can also be deleted
  static std::vector<int64_t> ParseLayout(const std::string& list_str,
                                          int64_t rank) {
    std::vector<int64_t> ints = ParseIntList(list_str);
    CHECK_EQ(ints.size(), rank) << list_str;
    std::set<int64_t> unique_ints;
    for (auto dim : ints) {
      CHECK_GE(dim, 0) << list_str;
      CHECK_LT(dim, rank) << list_str;
      unique_ints.insert(dim);
    }
    CHECK_EQ(unique_ints.size(), rank) << list_str;
    return ints;
  }

  LayoutMap layouts_;
};

}  // namespace

lazy_tensors::Shape MakeTorchTensorLayout(
    c10::ArrayRef<int64_t> dimensions, c10::ScalarType type) {
  lazy_tensors::Shape shape =
      lazy_tensors::ShapeUtil::MakeShapeWithDescendingLayout(type, dimensions);
  return shape;
}

lazy_tensors::Shape MakeArrayShapeFromDimensions(
    c10::ArrayRef<int64_t> dimensions, c10::ScalarType type, DeviceType device_type) {
  auto layout_ptr = LayoutManager::Get()->GetLayout(dimensions);
  if (layout_ptr != nullptr) {
    return lazy_tensors::ShapeUtil::MakeShapeWithLayout(type, dimensions, *layout_ptr);
  }
  return MakeTorchTensorLayout(dimensions, type);
}

}  // namespace torch_lazy_tensors
