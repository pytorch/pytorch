// included as 'TensorDenseOrSparse' in TensorDerived.cpp

IntList ${Tensor}::strides() const {
  return IntList(tensor->stride,dim());
}
Scalar ${Tensor}::localScalar() {
  int64_t numel = ${THTensor}_nElement(${state,}tensor);
  AT_CHECK(numel == 1,"localScalar() called on Tensor with ", numel, " elements");
  return Scalar(${to_at_type}(${THStorage}_get(${state,}tensor->storage, tensor->storageOffset)));
}
std::unique_ptr<Storage> ${Tensor}::storage() {
  auto storage = ${THTensor}_storage(${state,}tensor);
  ${THStorage}_retain(${state,}storage);
  return std::unique_ptr<Storage>(new ${Storage}(&type().get_context(), storage));
}
