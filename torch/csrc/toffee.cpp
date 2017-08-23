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

GraphProto* AttributeProto::add_graphs() {
  auto ptr = new GraphProto();
  graphs.emplace_back(ptr);
  return ptr;
}

}} // namespace toffee
