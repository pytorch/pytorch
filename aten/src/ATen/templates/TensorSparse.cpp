// included as 'TensorDenseOrSparse' in TensorDerived.cpp
IntList ${Tensor}::strides() const {
  runtime_error("Sparse tensors do not have strides.");
}
Scalar ${Tensor}::localScalar() {
  runtime_error("NYI localScalar() on sparse tensors.");
}
std::unique_ptr<Storage> ${Tensor}::storage() {
  runtime_error("storage() is not implemented for %s", type().toString());
}
