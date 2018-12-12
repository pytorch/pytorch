#include <torch/csrc/jit/batched/BatchTensor.h>

namespace torch { namespace jit {

BatchTensor::BatchTensor(at::Tensor data, at::Tensor mask, at::Tensor dims){
  if(data.dim() != mask.dim() || mask.dim() != dims.size(0) + 1){
    throw std::runtime_error("malformed MaskedBatch with data.dim(): "
      + std::to_string(data.dim()) + ", mask.dim(): " + std::to_string(mask.dim())
      + ", dims.size(0): " + std::to_string(dims.size(0)));
  }
  this->data = data;
  this->mask = mask;
  this->dims = dims;
}

BatchTensor::BatchTensor(at::Tensor data, int64_t batch_size){
  dims = at::empty(data.dim(), data.options().dtype(at::kByte));
  dims.fill_(0);
  std::vector<int64_t> sizes(data.dim() + 1, -1);
  sizes[0] = batch_size;
  this->data = data.unsqueeze(0).expand(sizes);
  std::vector<int64_t> mask_sizes(data.dim() + 1, 1);
  mask_sizes[0] = batch_size;
  mask = at::empty(mask_sizes, data.options().dtype(at::kByte));
  mask.fill_(1);
}

BatchTensor::BatchTensor(const std::vector<at::Tensor> datalist, at::Tensor dims) {
  auto bs = datalist.size();
  std::vector<int64_t> sizes(dims.size(0) + 1, 0), mask_sizes(dims.size(0) + 1, 0);
  sizes[0] = bs;
  mask_sizes[0] = bs;
  for(int64_t i = 1; i < dims.size(0) + 1; i++){
    for(auto x : datalist){
      sizes[i] = std::max(sizes[i], x.size(i));
    }
    mask_sizes[i] = *dims[i - 1].data<uint8_t>() ? sizes[i] : 1;
  }
  data = at::empty(sizes, datalist[0].options());
  data.fill_(0);
  mask = at::empty(mask_sizes, datalist[0].options().dtype(at::kByte));
  mask.fill_(0);
  for(std::size_t i = 0; i < datalist.size(); i++){
    auto data_item = data.narrow(0, i, 1);
    auto mask_item = mask.narrow(0, i, 1);
    for(int64_t j = 0; j < dims.size(0); j++){
      if(*dims[j].data<uint8_t>()){
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
  // calculate number of valid entries in dth dimension of data
  auto mask_sum = [](at::Tensor data, int d) -> int64_t{
    data = data.sum(d, /*keepdim=*/true);
    while(data.dim() >= 1)
      data = data[0];
    return *data.data<int64_t>();
  };
  for(int64_t i = 0; i < data.size(0); i++){
    auto data_tmp = data.narrow(0, i, 1);
    for(int64_t d = 0; d < dims.size(0); d++){
      if(*dims[d].data<uint8_t>()){
        data_tmp = data_tmp.narrow(d + 1, 0, mask_sum(mask[i], d));
      }
    }
    result.push_back(data_tmp);
  }
  return result;
}

void initBatchTensorBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  auto jit = m.def_submodule("_jit");
  py::class_<BatchTensor>(jit, "BatchTensor")
      .def(py::init<at::Tensor, at::Tensor, at::Tensor>())
      .def(py::init<at::Tensor, int64_t>())
      .def(py::init<std::vector<at::Tensor>, at::Tensor>())
      .def("examples", &BatchTensor::examples)
      .def("get_data", &BatchTensor::get_data)
      .def("get_mask", &BatchTensor::get_mask)
      .def("get_dims", &BatchTensor::get_dims);
}

}} // namespace torch::jit
