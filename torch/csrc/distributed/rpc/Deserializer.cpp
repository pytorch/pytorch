#include <torch/csrc/distributed/rpc/Deserializer.h>

namespace torch {
namespace distributed {
namespace rpc {

std::vector<at::IValue> Deserializer::readNext(
    std::istream& is, int64_t size) {
  std::vector<at::Tensor> tensor_table;
  torch::load(tensor_table, is);
  auto meta_tensor = std::move(tensor_table.back());
  tensor_table.pop_back();

  Unpickler unpickler(meta_tensor.storage().data(),
                      meta_tensor.numel(),
                      &tensor_table);

  return unpickler.parse_ivalue_list();
}

}
}
}
