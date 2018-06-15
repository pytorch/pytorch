#include "BatchTensorType.h"


BatchTensorType::BatchTensorType(at::Context* context)
: Type(context, /*is_variable_or_undefined=*/true) {}

at::ScalarType BatchTensorType::scalarType() const {
  return at::ScalarType::Undefined;
}

at::Backend BatchTensorType::backend() const {
  return at::Backend::Undefined;
}
bool BatchTensorType::is_cuda() const { return false; }
bool BatchTensorType::is_sparse() const { return false; }
bool BatchTensorType::is_distributed() const { return false; }

std::unique_ptr<at::Storage> BatchTensorType::storage() const {
  AT_ERROR("storage not defined for BatchTensorType");
}
std::unique_ptr<at::Storage> BatchTensorType::storage(size_t size) const {
  AT_ERROR("storage(size_t) not defined for BatchTensorType");
}
std::unique_ptr<at::Storage> BatchTensorType::storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const {
  AT_ERROR("storageFromBlob not defined for BatchTensorType");
}
std::unique_ptr<at::Storage> BatchTensorType::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  AT_ERROR("unsafeStorageFromTH not defined for BatchTensorType");
}
std::unique_ptr<at::Storage> BatchTensorType::storageWithAllocator(int64_t size, std::unique_ptr<at::Allocator> allocator) const {
  AT_ERROR("storageWithAllocator not defined for BatchTensorType");
}
at::Tensor BatchTensorType::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  AT_ERROR("unsafeTensorFromTH not defined for BatchTensorType");
}
std::unique_ptr<at::Generator> BatchTensorType::generator() const {
  AT_ERROR("generator not defined for BatchTensorType");
}


const char * BatchTensorType::toString() const {
  return BatchTensorType::typeString();
}
at::TypeID BatchTensorType::ID() const {
  return at::TypeID::Undefined;
}

size_t BatchTensorType::elementSizeInBytes() const {
  AT_ERROR("elementSizeInBytes not defined for BatchTensorType");
}

at::Type & BatchTensorType::toBackend(at::Backend b) const {
  if (b == at::Backend::Undefined) {
    return Type::toBackend(b);
  }
  AT_ERROR("toBackend not implemented for BatchTensorType to non-BatchTensorType");
}
at::Type & BatchTensorType::toScalarType(at::ScalarType s) const {
  if (s == at::ScalarType::Undefined) {
    return at::Type::toScalarType(s);
  }
  AT_ERROR("toScalarType not implemented for BatchTensorType to non-BatchTensorType");
}

const char * BatchTensorType::typeString() {
  return "BatchTensorType";
}

at::Tensor & BatchTensorType::s_copy_(at::Tensor & self, const at::Tensor & src, bool non_blocking) const {
  AT_ERROR("s_copy not defined for BatchTensorType");
}

at::Tensor & BatchTensorType::_s_copy_from(const at::Tensor & self, at::Tensor & dst, bool non_blocking) const {
  AT_ERROR("_s_copy_from not defined for BatchTensorType");
}

at::Tensor BatchTensorType::sigmoid(const at::Tensor & self) const{
  auto result_ = new BatchTensor();
  auto result = at::Tensor(result_, false);
  auto self_ = at::checked_cast_tensor<BatchTensor>(self.pImpl,"self",1, false);
  result_->mask = self_->mask;
  result_->dims = self_->dims;
  result_->data = self_->data.sigmoid();
  return result;
}

at::Tensor BatchTensorType::tanh(const at::Tensor & self) const{
  auto result_ = new BatchTensor();
  auto result = at::Tensor(result_, false);
  auto self_ = at::checked_cast_tensor<BatchTensor>(self.pImpl,"self",1, false);
  result_->mask = self_->mask;
  result_->dims = self_->dims;
  result_->data = self_->data.tanh();
  return result;
}

at::Tensor BatchTensorType::relu(const at::Tensor & self) const{
  auto result_ = new BatchTensor();
  auto result = at::Tensor(result_, false);
  auto self_ = at::checked_cast_tensor<BatchTensor>(self.pImpl,"self",1, false);
  result_->mask = self_->mask;
  result_->dims = self_->dims;
  result_->data = self_->data.relu();
  return result;
}

at::Tensor BatchTensorType::s_add(const at::Tensor & self, const at::Tensor & other, at::Scalar alpha) const {
    auto result_ = new BatchTensor();
    auto result = at::Tensor(result_, false);
    auto self_ = at::checked_cast_tensor<BatchTensor>(self.pImpl,"self",1, false);
    if(typeid(other.pImpl) == typeid(BatchTensor)){
      auto other_ = at::checked_cast_tensor<BatchTensor>(other.pImpl,"other",3, false);
      result_->data = self_->data.add(other_->data, alpha);
      result_->mask = self_->mask.mul(other_->mask);
      result_->dims = std::vector<bool>(self_->dims.size());
      for(std::size_t i = 0; i < self_->dims.size(); i++){
        result_->dims[i] = self_->dims[i] | other_->dims[i];
      }
    }
    else{
      // result_->data = self_->data.add(other.tensor, alpha);
      result_->mask = self_->mask;
      result_->dims = self_->dims;
    }
    return result;
}

at::Tensor BatchTensorType::add(const at::Tensor & self, at::Scalar other, at::Scalar alpha) const {
  auto result_ = new BatchTensor();
  auto result = at::Tensor(result_, false);
  auto self_ = at::checked_cast_tensor<BatchTensor>(self.pImpl,"self",1, false);
  result_->data = self_->data.add(other, alpha);
  result_->mask = self_->mask;
  result_->dims = self_->dims;
  return result;
}

at::Tensor BatchTensorType::matmul(const at::Tensor & self, const at::Tensor & other) const {
  auto result_ = new BatchTensor();
  auto result = at::Tensor(result_, false);
  auto self_ = at::checked_cast_tensor<BatchTensor>(self.pImpl,"self",1, false);
  auto other_ = at::checked_cast_tensor<BatchTensor>(other.pImpl,"other",3, false);
  auto d1 = self_->dims.size();
  auto d2 = other_->dims.size();
  auto data1 = self_->data * self_->mask.type_as(self_->data);
  auto data2 = other_->data * other_->mask.type_as(other_->data);
  if (d1 == 1){
    data1 = data1.unsqueeze(-2);
  }
  if (d2 == 2){
    data2 = data2.unsqueeze(-1);
  }
  result_->data = at::bmm(data1, data2);
  if (d1 == 1 && d2 == 1) {
    result_->data = result_->data.squeeze(-1).squeeze(-1);
    result_->mask = self_->mask.narrow(1, 0, 1).squeeze(-1);
    result_->dims = std::vector<bool>();
  }
  else if (d1 == 2 && d2 == 1) {
    result_->data = result_->data.squeeze(-1);
    result_->mask =
        at::bmm(self_->mask.narrow(2, 0, 1), other_->mask.narrow(1, 0, 1)
            .unsqueeze(-1)).squeeze(-1);
    result_->dims = std::vector<bool>(self_->dims.begin(), self_->dims.begin() + 1);
  }
  else if (d1 == 1 && d2 == 2) {
    result_->data = result_->data.squeeze(-2);
    result_->mask =
        at::bmm(self_->mask.narrow(1, 0, 1).unsqueeze(-2), other_->mask
            .narrow(1, 0, 1)).squeeze(-2);
    result_->dims = std::vector<bool>(other_->dims.begin() + 1, other_->dims.end());
  }
  else if (d1 == 2 and d2 == 2) {
    result_->mask = at::bmm(self_->mask.narrow(2, 0, 1), other_->mask.narrow(1, 0, 1));
    result_->dims = std::vector<bool>(self_->dims.begin(), self_->dims.begin() + 1);
    result_->dims.insert(result_->dims.end(), other_->dims.begin() + 1, other_->dims.end());
  }
  else {
    throw std::runtime_error("matmul not implemented with batches of 3+D tensors");
  }
  return result;
}

at::Tensor BatchTensorType::contiguous(const at::Tensor & self) const {
  auto result_ = new BatchTensor();
  auto result = at::Tensor(result_, false);
  auto self_ = at::checked_cast_tensor<BatchTensor>(self.pImpl,"self",1, false);
  result_->data = result_->data.contiguous();
  result_->mask = result_->mask.contiguous();
  result_->dims = self_->dims;
  return result;
}

// assumption: (sizes[i] == -1) === ith dimension is dynamic
at::Tensor BatchTensorType::view(const at::Tensor & self, at::IntList size) const {
  auto result_ = new BatchTensor();
  auto result = at::Tensor(result_, false);
  auto self_ = at::checked_cast_tensor<BatchTensor>(self.pImpl,"self",1, false);
  auto bs = self_->data.size(0);
  if (size[0] != 1 && size[0] != -1 && size[0] != bs){
    throw std::runtime_error("first dim in view must be 1, -1, or batch size");
  }
  std::vector<int64_t> data_sizes = {bs}, mask_sizes = {bs};
  result_->dims = std::vector<bool>(size.size() - 1, false);
  for (int i = 1; i < int(size.size()); i++) {
    data_sizes.push_back(size[i]);
    mask_sizes.push_back(size[i] == -1 ? self_->data.size(i) : 1);
    result_->dims[i - 1] = (size[i] == -1);
  }
  result_->data = self_->data.view(at::IntList(data_sizes));
  result_->mask = self_->mask.view(at::IntList(mask_sizes));
  return result;
}
