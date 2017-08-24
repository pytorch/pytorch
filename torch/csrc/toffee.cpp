#include "torch/csrc/toffee.h"

namespace torch { namespace toffee {

template <>
bool micropb_encode<std::string, nullptr>(pb_ostream_t *stream, std::string* arg) {
  return pb_encode_string(stream, reinterpret_cast<const pb_byte_t *>(arg->c_str()), arg->size());
}
// NB: Overloads don't work so great for signed variables.  Hope this doesn't
// come up!
template <>
bool micropb_encode<int64_t, nullptr>(pb_ostream_t *stream, int64_t* arg) {
  // Yes, this looks dodgy, and yes, this is what the docs say to do:
  // https://jpa.kapsi.fi/nanopb/docs/reference.html#pb-encode-varint
  return pb_encode_varint(stream, *reinterpret_cast<uint64_t*>(arg));
}
template <>
bool micropb_encode<float, nullptr>(pb_ostream_t *stream, float* arg) {
  return pb_encode_fixed32(stream, static_cast<void*>(arg));
}
template <>
bool micropb_encode<double, nullptr>(pb_ostream_t *stream, double* arg) {
  return pb_encode_fixed64(stream, static_cast<void*>(arg));
}

// TODO: I'm not entirely sure why this can't be in the header...
bool micropb_callback_tensor(pb_ostream_t *stream, const pb_field_t *field, void * const *arg) {
  at::Tensor* t = static_cast<at::Tensor*>(*arg);
  JIT_ASSERT(t->type().scalarType() == at::kFloat);
  JIT_ASSERT(t->is_contiguous());
  // TODO: Generalize beyond float
  // Packed array format!
  pb_encode_tag(stream, PB_WT_STRING, toffee_TensorProto_float_data_tag);
  static_assert(sizeof(float) == 4, "float is not four bytes");
  pb_encode_varint(stream, sizeof(float) * t->numel()); // number of bytes to write
  // TODO: If you have a better way of doing this, I'm all ears.
  float *addr = t->data<float>();
  for (float *p = addr; p < addr + t->numel(); p++) {
    if (!pb_encode_fixed32(stream, static_cast<void*>(p))) return false;
  }
  return true;
}

GraphProto* AttributeProto::add_graphs() {
  auto ptr = new GraphProto();
  graphs.emplace_back(ptr);
  return ptr;
}

}} // namespace toffee
