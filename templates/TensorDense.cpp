// included as 'TensorDenseOrSparse' in TensorDerived.cpp

IntList ${Tensor}::strides() {
  return IntList(reinterpret_cast<int64_t*>(tensor->stride),dim());
}
Scalar ${Tensor}::localScalar() {
  AT_ASSERT(isScalar(),"localScalar() called on Tensor with %d dims",sizes().size());
  return Scalar(${to_at_half}(${THTensor}_get1d(${state,}tensor, 0)));
}
void ${Tensor}::assign_(Scalar s) {
  AT_ASSERT(isScalar(),"assign_() called on Tensor with %d dims",sizes().size());
  ${THTensor}_set1d(${state,}tensor, 0,${to_th_half}(s.to${ScalarName}()));
}
