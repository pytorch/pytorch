// included as 'TensorDenseOrSparse' in TensorDerived.cpp
IntList ${Tensor}::strides() const {
  runtime_error("Sparse tensors do not have strides.");
}
Scalar ${Tensor}::localScalar() {
  runtime_error("NYI localScalar() on sparse tensors.");
}
void ${Tensor}::assign_(Scalar s) {
  runtime_error("NYI assign_() on sparse tensors.");
}
