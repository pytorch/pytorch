// included as 'TensorDenseOrSparse' in TensorDerived.cpp

std::unique_ptr<Storage> ${Tensor}::storage() {
  auto storage = THTensor_getStoragePtr(tensor);
  THStorage_retain(storage);
  return std::unique_ptr<Storage>(new Storage(storage));
}
