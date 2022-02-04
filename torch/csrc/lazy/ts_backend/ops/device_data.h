#pragma once

#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch {
namespace lazy {

class TORCH_API DeviceData : public TsNode {
 public:
  static std::vector<std::vector<std::shared_ptr<BackendData>>> backend_data_storage;
  static size_t backend_data_current_row;

  explicit DeviceData(std::shared_ptr<BackendData> data);

  std::string ToString() const override;

  const std::shared_ptr<BackendData>& data() const {
    // This is used when lowering BackendData into TSIR
    return backend_data_storage[row_][column_];
  }

  size_t Column() const {
    return column_;
  }

  static const DeviceData* Cast(const Node* node);

  static std::shared_ptr<BackendData>& GetBackendData(size_t row, size_t column) {
    TORCH_CHECK(row < backend_data_storage.size() &&
      column < backend_data_storage[row].size());
    return backend_data_storage[row][column];
  }

  static void AdvanceToNextRow();

  static void ClearRow(size_t row);

 private:
  size_t row_;
  size_t column_;
};

} // namespace lazy
} // namespace torch
