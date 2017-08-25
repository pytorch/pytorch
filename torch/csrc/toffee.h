#pragma once

#include "torch/csrc/toffee.pb.h"
#include "torch/csrc/jit/assert.h"

#include <pb_encode.h>
#include <ATen/ATen.h>

// #include <toffee/schema.h>
// #include <google/protobuf/text_format.h>
// #include <google/protobuf/io/zero_copy_stream_impl.h>

#include <vector>
#include <fstream>
#include <memory>

namespace torch { namespace toffee {

// Why do we need vectors of unique pointers?  A Google-style C++ Protobuf API
// returns raw pointers T* which are expected to stay valid as long as the
// enclosing protobuf is live.  However, if we store T directly in a vector, if
// the vector ever resizes (which it may, because we don't know a priori how
// many elements are in the vector) all of these pointers will be invalidated.
// Thus, up-front, we have to give them permanent, dynamically allocated
// addresses.
template<typename T>
using unique_vector = std::vector<std::unique_ptr<T>>;

// Helper function for encoding inside callbacks
template<typename T, const pb_field_t* Field>
bool micropb_encode(pb_ostream_t *stream, T* arg) {
  static_assert(Field != nullptr, "no overload in micropb_encode");
  return pb_encode_submessage(stream, Field, static_cast<void*>(&arg->proto));
}
template <> bool micropb_encode<std::string, nullptr>(pb_ostream_t *stream, std::string* arg);
template <> bool micropb_encode<int64_t, nullptr>(pb_ostream_t *stream, int64_t* arg);
template <> bool micropb_encode<float, nullptr>(pb_ostream_t *stream, float* arg);
template <> bool micropb_encode<double, nullptr>(pb_ostream_t *stream, double* arg);
// NB: If we ever add support for signed protobuf integers, we'll need a special
// wrapper, since we can't overload over them (they look the same from C++ side)

// Callback functions of type pb_callback_t.

// Write out a single protobuf field inside a message
template<typename T, const pb_field_t* Field>
bool micropb_callback(pb_ostream_t *stream, const pb_field_t *field, void * const *arg) {
  if (!pb_encode_tag_for_field(stream, field)) return false;
  if (!micropb_encode<T, Field>(stream, static_cast<T*>(*arg))) return false;
  return true;
}

// Write out a repeated protobuf field inside a message
template<typename T, const pb_field_t* Field>
bool micropb_callback_list(pb_ostream_t *stream, const pb_field_t *field, void * const *arg) {
  std::vector<std::unique_ptr<T>>* vals = static_cast<std::vector<std::unique_ptr<T>>*>(*arg);
  for (std::unique_ptr<T>& val : *vals) {
    auto ptr = static_cast<void*>(val.get());
    if (!micropb_callback<T, Field>(stream, field, &ptr)) return false;
  }
  return true;
}

bool micropb_callback_tensor(pb_ostream_t *stream, const pb_field_t *field, void * const *arg);

// MicroProto helper class
template<typename T>
struct MicroProto {
  // The actual nanopb generated protobuf struct we are filling.
  T proto;

  // The constructor takes the protobuf struct by value for initialization
  // (since it is a C-style struct).  In the constructor you're
  // expected to call this with something like toffee_TensorProto_init_default
  MicroProto(T proto) : proto(proto) {}

  // Usage:
  //    std::string owning_slot;
  //    proto.string_field = string(&owning_slot, value_to_set)
  //
  // This function takes a string 's' and copies it into the
  // owning slot specified by 'slot'.  It then returns a callback
  // intended to be assigned into the particular protobuf field.
  // The employed callback reads out the string from owning
  // slot and writes it out to the protobuf.
  //
  // You should call this function IN THE SETTER METHOD, because
  // the no-op callback is different from a callback with an empty
  // string: in the former case, the field is absent; in the latter,
  // the field is present but an empty string.
  pb_callback_t string(std::string* slot, const std::string& s) {
    *slot = s; // copy construct
    pb_callback_t r;
    r.funcs.encode = &micropb_callback<std::string, nullptr>;
    r.arg = static_cast<void*>(slot);
    return r; // RVO
  }

  // NB: Needs to have been heap allocated from the get-go,
  // because we're going to take over ownership

  // Usage:
  //    unique_vector<ElemType> owning_slot;
  //    proto.list_field = list<ElemType>(&owning_slot)
  //
  // This function returns a callback intended to be
  // assigned into a particular protobuf field.  The employed
  // callback reads out the vector of elements from the owning
  // slot and writes the entries into the protobuf.
  //
  // You should call this function IN THE CONSTRUCTOR, because
  // the no-op callback is equivalent to a callback with an empty
  // list.  (While it's harmless to call this in the setter, but
  // a bit wasteful.)
  template<typename S, const pb_field_t* Field = nullptr>
  pb_callback_t list(unique_vector<S>* slot) {
    pb_callback_t r;
    r.funcs.encode = &micropb_callback_list<S, Field>;
    r.arg = static_cast<void*>(slot);
    return r; // RVO
  }

  pb_callback_t tensor(at::Tensor* slot, const at::Tensor& t) {
    *slot = t; // copy construct
    pb_callback_t r;
    r.funcs.encode = &micropb_callback_tensor;
    r.arg = static_cast<void*>(slot);
    return r; // RVO
  }
};

// TODO: add more of these as necessary
const toffee_TensorProto_DataType TensorProto_DataType_FLOAT = toffee_TensorProto_DataType_FLOAT;

// C++ wrappers which simulate the Google C++ Protobuf API
//
// These are NOT COMPLETE wrappers. If you find something is missing, add it!

class AttributeProto;
class NodeProto;
class GraphProto;
class TensorProto;

class TensorProto : public MicroProto<toffee_TensorProto> {
private:
  std::string name;
  unique_vector<int64_t> dims;
  at::Tensor tensor_data;
public:
  TensorProto() : MicroProto(toffee_TensorProto_init_default) {
    proto.dims       = list<int64_t>(&dims);
  }
  void set_name(const std::string& s) { proto.name = string(&name, s); }
  void add_dims(int64_t d) { dims.emplace_back(new int64_t(d)); }
  void add_tensor(const at::Tensor& t) {
    if (t.type().scalarType() == at::kFloat) {
      proto.raw_data = tensor(&tensor_data, t);
    } else {
      JIT_ASSERTM(0, "non-float tensors not supported yet");
    }
  }
  void set_data_type(toffee_TensorProto_DataType t) { proto.has_data_type = true; proto.data_type = t; }
};

class AttributeProto : public MicroProto<toffee_AttributeProto> {
private:
  std::string name;
  std::string s;
  unique_vector<float> floats;
  unique_vector<int64_t> ints;
  unique_vector<std::string> strings;
  unique_vector<TensorProto> tensors;
  unique_vector<GraphProto> graphs;
public:
  AttributeProto() : MicroProto(toffee_AttributeProto_init_default) {
    proto.floats  = list<float>(&floats);
    proto.ints    = list<int64_t>(&ints);
    proto.strings = list<std::string>(&strings);
    proto.tensors = list<TensorProto, toffee_TensorProto_fields>(&tensors);
    proto.graphs  = list<GraphProto, toffee_GraphProto_fields>(&graphs);
  }
  void set_name(const std::string& s) { proto.name = string(&name, s); }
  void set_f(float f) { proto.has_f = true; proto.f = f; }
  void set_i(int64_t i) { proto.has_i = true; proto.i = i; }
  void set_s(std::string s_) { proto.s = string(&s, s_); }
  void add_floats(float f) { floats.emplace_back(new float(f)); }
  void add_ints(int64_t i) { ints.emplace_back(new int64_t(i)); }
  void add_strings(std::string s) { strings.emplace_back(new std::string(s)); }
  TensorProto* add_tensors() {
    auto ptr = new TensorProto();
    tensors.emplace_back(ptr);
    return ptr;
  }
  GraphProto* add_graphs();
};

class NodeProto : public MicroProto<toffee_NodeProto> {
private:
  std::string op_type;
  unique_vector<std::string> inputs;
  unique_vector<std::string> outputs;
  unique_vector<AttributeProto> attributes;
public:
  NodeProto() : MicroProto(toffee_NodeProto_init_default) {
    proto.input     = list<std::string>(&inputs);
    proto.output    = list<std::string>(&outputs);
    proto.attribute = list<AttributeProto, toffee_AttributeProto_fields>(&attributes);
  }
  void add_input(const std::string& s) { inputs.emplace_back(new std::string(s)); }
  void clear_input() { inputs.clear(); }
  void add_output(const std::string& s) { outputs.emplace_back(new std::string(s)); }
  void clear_output() { outputs.clear(); }
  AttributeProto* add_attribute() {
    auto ptr = new AttributeProto();
    attributes.emplace_back(ptr);
    return ptr;
  }
  void set_op_type(const std::string& s) { proto.op_type= string(&op_type, s); }
};

class GraphProto : public MicroProto<toffee_GraphProto> {
private:
  std::string name;
  unique_vector<std::string> inputs;
  unique_vector<std::string> outputs;
  unique_vector<NodeProto> nodes;
  unique_vector<TensorProto> initializers;
public:
  GraphProto() : MicroProto(toffee_GraphProto_init_default) {
    proto.input  = list<std::string>(&inputs);
    proto.output = list<std::string>(&outputs);
    proto.node   = list<NodeProto, toffee_NodeProto_fields>(&nodes);
    proto.initializer = list<TensorProto, toffee_TensorProto_fields>(&initializers);
  }
  void set_name(const std::string& s) { proto.name = string(&name, s); }
  void add_input(const std::string& s) { inputs.emplace_back(new std::string(s)); }
  std::string input(size_t i) { return *inputs.at(i); }
  void add_output(const std::string& s) { outputs.emplace_back(new std::string(s)); }
  NodeProto* add_node() {
    auto ptr = new NodeProto();
    nodes.emplace_back(ptr);
    return ptr;
  }
  TensorProto* add_initializer() {
    auto ptr = new TensorProto();
    initializers.emplace_back(ptr);
    return ptr;
  }
};

}} // namespace torch::toffee
