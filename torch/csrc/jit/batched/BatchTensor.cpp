#include "BatchTensor.h"

namespace torch { namespace jit {

BatchTensor::BatchTensor(at::Tensor data, at::Tensor mask, std::vector<bool> dims){
  if(data.dim() != mask.dim() || mask.dim() != int64_t(dims.size()) + 1){
    throw std::runtime_error("malformed MaskedBatch with data: "
      + std::string(data.toString()) + " mask: " + mask.toString());
  }
  this->data = data;
  this->mask = mask;
  this->dims = dims;
}

BatchTensor::BatchTensor(std::vector<at::Tensor> datalist, std::vector<bool> dims) {
  auto bs = datalist.size();
  std::vector<int64_t> sizes(dims.size() + 1, 0), mask_sizes(dims.size() + 1, 0);
  sizes[0] = bs;
  mask_sizes[0] = bs;
  for(std::size_t i = 1; i < dims.size() + 1; i++){
    for(auto x : datalist){
      sizes[i] = std::max(sizes[i], x.size(i));
    }
    mask_sizes[i] = dims[i - 1] ? sizes[i] : 1;
  }
  data = datalist[0].type().zeros(sizes);
  mask = datalist[0].type().toScalarType(at::kByte).zeros(mask_sizes);
  for(std::size_t i = 0; i < datalist.size(); i++){
    auto data_item = data.narrow(0, i, 1);
    auto mask_item = mask.narrow(0, i, 1);
    for(std::size_t j = 0; j < dims.size(); j++){
      if(dims[j]){
        data_item = data_item.narrow(j + 1, 0, datalist[i].size(j + 1));
        mask_item = mask_item.narrow(j + 1, 0, datalist[i].size(j + 1));
      }
    }
    data_item += datalist[i];
    mask_item.fill_(1);
  }
  this->dims = dims;
}

std::vector<at::Tensor> BatchTensor::examples() {
  std::vector<at::Tensor> result;
  auto mask_sum = [](at::Tensor a) -> int64_t{
    while(a.dim() >= 1)
      a = a[0];
    return *a.toLongData();
  };
  for(int64_t i = 0; i < data.size(0); i++){
    auto data_tmp = data.narrow(0, i, 1);
    for(std::size_t d = 0; d < dims.size(); d++){
      if(dims[d]){
        data_tmp = data_tmp.narrow(d + 1, 0, mask_sum(mask[i].sum(d, true)));
      }
    }
    result.push_back(data_tmp);
  }
  return result;
}

void initBatchTensorBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  py::class_<BatchTensor>(m, "BatchTensor")
      .def(py::init<at::Tensor, at::Tensor, std::vector<bool>>())
      .def(py::init<std::vector<at::Tensor>, std::vector<bool>>())
      .def("examples", &BatchTensor::examples);
}

}} // namespace torch::jit
