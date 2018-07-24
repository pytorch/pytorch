// included as 'TensorDenseOrSparse' in TensorDerived.cpp

Scalar ${Tensor}::localScalar() {
  int64_t numel = ${THTensor}_nElement(${state,}tensor);
  AT_CHECK(numel == 1,"a Tensor with ", numel, " elements cannot be converted to Scalar");
  return Scalar(${to_at_type}(${THStorage}_get(${state,} THTensor_getStoragePtr(tensor), tensor->storage_offset())));
}
std::unique_ptr<Storage> ${Tensor}::storage() {
  auto storage = THTensor_getStoragePtr(tensor);
  THStorage_retain(storage);
  return std::unique_ptr<Storage>(new ${Storage}(storage));
}
