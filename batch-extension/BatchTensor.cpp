#include "BatchTensor.h"


static BatchTensorType& batch_tensor_singleton() {
  static BatchTensorType value(&at::globalContext());
  return value;
}

BatchTensor::BatchTensor(): TensorImpl(&batch_tensor_singleton()) {}

BatchTensor::BatchTensor(at::Tensor data, at::Tensor mask, std::vector<bool> dims):
TensorImpl(&batch_tensor_singleton()){
  if(data.dim() != mask.dim() || mask.dim() != dims.size() + 1){
    throw std::runtime_error("malformed MaskedBatch with " + std::string(data.toString()) + mask.toString());
  }
  data = data;
  mask = mask;
  dims = dims;
}

BatchTensor::BatchTensor(std::vector<at::Tensor> datalist, std::vector<bool> dims):
TensorImpl(&batch_tensor_singleton()){
  auto bs = datalist.size();
  std::vector<int64_t> sizes(dims.size() + 1, 0), mask_sizes(dims.size() + 1, 0);
  sizes[0] = bs;
  mask_sizes[0] = bs;
  for(std::size_t i = 1; i < dims.size() + 1; i++){
    for(auto x : datalist){
      sizes[i] = std::max(sizes[i], x.size(i));
    }
    mask_sizes[i] = dims[i] ? sizes[i] : 1;
  }
  this->data = at::zeros(datalist[0].type(), sizes);
  this->mask = at::zeros(datalist[0].type().toScalarType(at::kByte), mask_sizes);
  for(std::size_t i = 0; i < datalist.size(); i++){
    auto data_item = this->data.narrow(0, i, 1);
    auto mask_item = this->mask.narrow(0, i, 1);
    for(std::size_t j = 0; j < dims.size(); j++){
      if(dims[j]){
        data_item = data_item.narrow(j + 1, 0, datalist[i].size(j + 1));
        mask_item = data_item.narrow(j + 1, 0, datalist[i].size(j + 1));
      }
    }
    data_item.fill_(0);
    data_item += datalist[i];
    mask_item.fill_(0);
  }
  this->dims = dims;
}

const char * BatchTensor::toString() const{
  return "BatchTensor";
}

const char * BatchTensor::typeString() {
  return "BatchTensorType";
}


std::vector<at::Tensor> BatchTensor::examples () const {
  std::vector<at::Tensor> result;
  auto mask_sum = [](at::Tensor a) -> int64_t{
    while(a.dim() >= 1)
      a = a[0];
    return *a.toLongData();
  };
  for(std::size_t i = 0; i < this->data.size(0); i++){
    auto data = this->data.narrow(0, i, 1);
    for(std::size_t d = 0; d < dims.size(); d++){
      if(dims[d]){
        data = data.narrow(d + 1, 0, mask_sum(mask[i].sum(d, true)));
      }
    }
    result.push_back(data);
  }
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<BatchTensor>(m, "BatchTensor")
  .def(py::init<>())
  .def(py::init<at::Tensor, at::Tensor, std::vector<bool>>())
  .def(py::init<std::vector<at::Tensor>, std::vector<bool>>())
  .def("examples", &BatchTensor::examples)
  .def("__repr__", &BatchTensor::toString);

  py::class_<BatchTensorType>(m, "BatchTensorType")
  .def(py::init<at::Context*>())
  .def("s_add", &BatchTensorType::s_add)
  .def("add", &BatchTensorType::add)
  .def("sigmoid", &BatchTensorType::sigmoid)
  .def("tanh", &BatchTensorType::tanh)
  .def("relu", &BatchTensorType::relu)
  .def("matmul", &BatchTensorType::matmul)
  .def("contiguous", &BatchTensorType::contiguous)
  .def("view", &BatchTensorType::view);
}
