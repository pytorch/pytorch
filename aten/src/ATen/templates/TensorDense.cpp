// included as 'TensorDenseOrSparse' in TensorDerived.cpp

IntList ${Tensor}::strides() const {
  int64_t d = tensor->nDimension;
  if (d != 0) {
    return IntList(reinterpret_cast<int64_t*>(tensor->stride),dim());
  } else {
    return IntList(kEmptyStrides);
  }
}
Scalar ${Tensor}::localScalar() {
  int64_t numel = ${THTensor}_nElement(${state,}tensor);
  AT_ASSERT(numel == 1,"localScalar() called on Tensor with %" PRId64 " elements",numel);
  return Scalar(${to_at_type}(${THStorage}_get(${state,}tensor->storage, tensor->storageOffset)));
}
std::unique_ptr<Storage> ${Tensor}::storage() {
  auto storage = ${THTensor}_storage(${state,}tensor);
  ${THStorage}_retain(${state,}storage);
  return std::unique_ptr<Storage>(new ${Storage}(&type().get_context(), storage));
}
