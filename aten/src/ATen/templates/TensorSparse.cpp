// included as 'TensorDenseOrSparse' in TensorDerived.cpp
IntList ${Tensor}::strides() const {
 AT_ERROR("Sparse tensors do not have strides.");
}
Scalar ${Tensor}::localScalar() {
 AT_ERROR("NYI localScalar() on sparse tensors.");
}
std::unique_ptr<Storage> ${Tensor}::storage() {
  AT_ERROR("storage() is not implemented for %s", type().toString());
}
