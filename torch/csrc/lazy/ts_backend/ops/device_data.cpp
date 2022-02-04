#include <torch/csrc/lazy/ts_backend/ops/device_data.h>

#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>

#include <sstream>

namespace torch {
namespace lazy {

std::vector<std::vector<std::shared_ptr<BackendData>>> DeviceData::backend_data_storage{{}};

size_t DeviceData::backend_data_current_row = 0;

DeviceData::DeviceData(std::shared_ptr<BackendData> data)
    : TsNode(
          ltc_device_data,
          data->shape(),
          /*num_outputs=*/1,
          /*Use column_ as hash_seed*/
          backend_data_storage[backend_data_current_row].size()),
      row_(backend_data_current_row),
      column_(backend_data_storage[row_].size()) {
    //std::cout << std::endl << "Pushing into backend_data_storage[" << row_ << "][" << column_ << "]";
    backend_data_storage[row_].push_back(std::move(data));
}

std::string DeviceData::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", device=" << data()->device();
  return ss.str();
}

const DeviceData* DeviceData::Cast(const Node* node) {
  return NodeCast<DeviceData>(node, ltc_device_data);
}

void DeviceData::AdvanceToNextRow() {
  // Bump up backend_data_current_row so that the next tracing iteration
  // can write to a different row
  DeviceData::backend_data_storage.push_back({});
  DeviceData::backend_data_current_row++;
}

void DeviceData::ClearRow(size_t row) {
  if (row < backend_data_storage.size()) {
    DeviceData::backend_data_storage[row].clear();
  }
}

} // namespace lazy
} // namespace torch
