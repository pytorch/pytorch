#include <torch/csrc/WindowsTorchApiMacro.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/pickle.h>
#include <torch/csrc/jit/pickler.h>


namespace torch {
namespace jit {

void pickle(
    std::function<void(const char*, size_t)> writer,
    const IValue& ivalue,
    std::vector<at::Tensor>* tensor_table) {
  Pickler pickler(std::move(writer), tensor_table);

  if (tensor_table == nullptr) {
    // No tensor table provided, so tensors will be stored directly in the blob.
    // Add torch.save metadata so these tensors can be de-serialized later
    pickler.torchSaveStart();
  }

  pickler.protocol();
  pickler.pushIValue(ivalue);
  pickler.stop();

  if (tensor_table == nullptr) {
    // No tensor table provided, so tensors will be stored directly in the blob.
    // Add torch.save metadata so these tensors can be de-serialized later
    pickler.torchSaveStop();
  }
}

std::vector<char> pickle(
    const IValue& ivalue,
    std::vector<at::Tensor>* tensor_table) {
  std::vector<char> data;

  pickle(
      [&](const char* bytes, size_t len) {
        data.insert(data.end(), bytes, bytes + len);
      },
      ivalue,
      tensor_table);

  return data;
}

} // namespace jit
} // namespace torch
