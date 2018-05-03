#include "torch/csrc/jit/import.h"
#include "torch/csrc/onnx/onnx.pb.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/utils/functional.h"

#include <ATen/ATen.h>

#include <unordered_map>
#include <vector>
#include <string>

#include "third_party/nanopb/pb_decode.h"

namespace torch { namespace jit {

namespace {

// Deserialized data

struct Tensor_ {
  std::vector<int64_t> dims;
  std::vector<uint8_t> raw_data;
  onnx_TensorProto_DataType data_type;
};

struct AttributeValue_ {
  std::string name;
  onnx_AttributeProto_AttributeType type;
  double f;
  int64_t i;
  std::string s;
  Tensor_ t;
  std::string g;
  std::vector<double> fs;
  std::vector<int64_t> is;
  std::vector<std::string> ss;
  std::vector<Tensor_> ts;
  std::vector<std::string> gs;
};

struct Value_ {
  std::string name;
};

struct Node_ {
  std::string op_type;
  std::string domain;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::vector<AttributeValue_> attrs;
};

struct Graph_ {
  std::vector<Value_> inputs;
  std::vector<Value_> outputs;
  std::vector<Node_> nodes;
  std::vector<Tensor_> initializers;
};

struct Model_ {
  Graph_ graph;
};


// Readers

struct ReaderBase {
  ReaderBase() {}
  ReaderBase(pb_callback_t& cb) {
    initialize_callback(cb);
  }

  void initialize_callback(pb_callback_t& cb) {
    cb.funcs.decode = ReaderBase::decode;
    cb.arg = this;
  }

  virtual void decode(pb_istream_t *stream) = 0;

  static bool decode(pb_istream_t *stream, const pb_field_t *, void **_self) {
    ReaderBase* self = *reinterpret_cast<ReaderBase* const *>(_self);
    self->decode(stream);
    return true;
  }
};


template<typename T>
struct Reader : ReaderBase {};

template<typename T>
struct Reader<std::vector<T>> : Reader<T> {
  Reader(pb_callback_t& cb) : Reader<T>(cb) {}
  // Decode is going to be called repeatedly from the callback
  // (registered in the parent class constructor) each time an
  // element is encountered. So all we do is relay the decoding
  // through the parent class decode and push the result, every
  // time this decode is called.
  virtual void decode(pb_istream_t *stream) override {
    Reader<T>::decode(stream);
    values.push_back(std::move(Reader<T>::value));
  }
  std::vector<T> values;
};

template<>
struct Reader<std::string> : ReaderBase {
  Reader(pb_callback_t& cb) : ReaderBase(cb) {}
  virtual void decode(pb_istream_t *stream) override {
    // For string and bytes, the length value has already been
    // parsed, and is available at stream->bytes_left.
    std::vector<uint8_t> res(stream->bytes_left);
    if (!pb_read(stream, res.data(), stream->bytes_left)) {
      throw std::runtime_error("Decoding failed");
    }
    value.assign(res.begin(), res.end());
  }
  std::string value;
};

template<>
struct Reader<double> : ReaderBase {
  Reader(pb_callback_t& cb) : ReaderBase(cb) {}
  virtual void decode(pb_istream_t *stream) override {
    if (!pb_decode_fixed32(stream, &value)) {
      throw std::runtime_error("Decoding failed");
    }
  }
  double value;
};

template<>
struct Reader<int64_t> : ReaderBase {
  Reader(pb_callback_t& cb) : ReaderBase(cb) {}
  virtual void decode(pb_istream_t *stream) override {
    if (!pb_decode_varint(stream, reinterpret_cast<uint64_t*>(&value))) {
      throw std::runtime_error("Decoding failed");
    }
  }
  int64_t value;
};

template<>
struct Reader<std::vector<uint8_t>> : ReaderBase {
  Reader(pb_callback_t& cb) : ReaderBase(cb) {}
  virtual void decode(pb_istream_t *stream) override {
    // For string and bytes, the length value has already been
    // parsed, and is available at stream->bytes_left.
    value.resize(stream->bytes_left);
    if (!pb_read(stream, value.data(), stream->bytes_left)) {
      throw std::runtime_error("Decoding failed");
    }
  }
  std::vector<uint8_t> value;
};

template<>
struct Reader<Tensor_> : ReaderBase {
  Reader()
    : proto(onnx_TensorProto_init_default)
    , dims_reader(proto.dims)
    , raw_data_reader(proto.raw_data)
  {}

  Reader(pb_callback_t& cb)
    : Reader() { initialize_callback(cb); }

  virtual void decode(pb_istream_t *stream) override {
    if (!pb_decode(stream, onnx_TensorProto_fields, &proto)) {
      throw std::runtime_error("Decoding failed");
    }

    value.dims = std::move(dims_reader.values);
    value.raw_data = std::move(raw_data_reader.value);
    value.data_type = proto.data_type;
  }

  onnx_TensorProto proto;
  Reader<std::vector<int64_t>> dims_reader;
  Reader<std::vector<uint8_t>> raw_data_reader;
  Tensor_ value;
};

template<>
struct Reader<AttributeValue_> : ReaderBase {
  Reader()
    : proto(onnx_AttributeProto_init_default)
    , name_reader(proto.name)
    , str_reader(proto.s)
    , tensor_reader(proto.t)
    , graph_reader(proto.g)
    , floats_reader(proto.floats)
    , ints_reader(proto.ints)
    , strings_reader(proto.strings)
    , tensors_reader(proto.tensors)
    , graphs_reader(proto.graphs) {}

  Reader(pb_callback_t& cb)
    : Reader() { initialize_callback(cb); }

  virtual void decode(pb_istream_t *stream) override {
    if (!pb_decode(stream, onnx_AttributeProto_fields, &proto)) {
      throw std::runtime_error("Decoding failed");
    }

    value.name = std::move(name_reader.value);
    value.type = proto.type;
    value.f = proto.f;
    value.i = proto.i;
    value.s = std::move(str_reader.value);
    value.t = std::move(tensor_reader.value);
    value.g = std::move(graph_reader.value);
    value.fs = std::move(floats_reader.values);
    value.is = std::move(ints_reader.values);
    value.ss = std::move(strings_reader.values);
    value.ts = std::move(tensors_reader.values);
    value.gs = std::move(graphs_reader.values);
  }

  onnx_AttributeProto proto;
  Reader<std::string> name_reader;
  Reader<std::string> str_reader;
  Reader<Tensor_> tensor_reader;
  Reader<std::string> graph_reader;
  Reader<std::vector<double>> floats_reader;
  Reader<std::vector<int64_t>> ints_reader;
  Reader<std::vector<std::string>> strings_reader;
  Reader<std::vector<Tensor_>> tensors_reader;
  Reader<std::vector<std::string>> graphs_reader;
  AttributeValue_ value;
};

template<>
struct Reader<Value_> : ReaderBase {
  Reader()
    : proto(onnx_ValueInfoProto_init_default)
    , name_reader(proto.name) {}
  Reader(pb_callback_t& cb)
    : Reader() { initialize_callback(cb); }

  virtual void decode(pb_istream_t *stream) override {
    if (!pb_decode(stream, onnx_ValueInfoProto_fields, &proto)) {
      throw std::runtime_error("Decoding failed");
    }

    value.name = std::move(name_reader.value);
  }

  onnx_ValueInfoProto proto;
  Reader<std::string> name_reader;
  Value_ value;
};


template<>
struct Reader<Node_> : ReaderBase {
  Reader()
    : proto(onnx_NodeProto_init_default)
    , op_type_reader(proto.op_type)
    , domain_reader(proto.domain)
    , inputs_reader(proto.input)
    , outputs_reader(proto.output)
    , attrs_reader(proto.attribute)
  {}
  Reader(pb_callback_t& cb)
    : Reader() { initialize_callback(cb); }

  virtual void decode(pb_istream_t *stream) override {
    if (!pb_decode(stream, onnx_NodeProto_fields, &proto)) {
      throw std::runtime_error("Decoding failed");
    }

    value.op_type = std::move(op_type_reader.value);
    value.domain = std::move(domain_reader.value);
    value.inputs = std::move(inputs_reader.values);
    value.outputs = std::move(outputs_reader.values);
    value.attrs = std::move(attrs_reader.values);
  }

  onnx_NodeProto proto;
  Reader<std::string> op_type_reader;
  Reader<std::string> domain_reader;
  Reader<std::vector<std::string>> inputs_reader;
  Reader<std::vector<std::string>> outputs_reader;
  Reader<std::vector<AttributeValue_>> attrs_reader;
  Node_ value;
};


template<>
struct Reader<Graph_> : ReaderBase {
  Reader()
    : proto(onnx_GraphProto_init_default)
    , input_reader(proto.input)
    , output_reader(proto.output)
    , node_reader(proto.node)
    , initializer_reader(proto.initializer)
  {}
  Reader(pb_callback_t& cb)
    : Reader() { initialize_callback(cb); }

  virtual void decode(pb_istream_t *stream) override {
    if (!pb_decode(stream, onnx_GraphProto_fields, &proto)) {
      throw std::runtime_error("Decoding failed");
    }

    value.inputs = std::move(input_reader.values);
    value.outputs = std::move(output_reader.values);
    value.nodes = std::move(node_reader.values);
    value.initializers = std::move(initializer_reader.values);
  }

  static Graph_ read(pb_istream_t *stream) {
    Reader<Graph_> reader;
    reader.decode(stream);
    return reader.value;
  }

  onnx_GraphProto proto;
  Reader<std::vector<Value_>> input_reader;
  Reader<std::vector<Value_>> output_reader;
  Reader<std::vector<Node_>> node_reader;
  Reader<std::vector<Tensor_>> initializer_reader;
  Graph_ value;
};


template<>
struct Reader<Model_> : ReaderBase {
  Reader()
    : proto(onnx_ModelProto_init_default)
    , graph_reader(proto.graph) {}
  Reader(pb_callback_t& cb)
    : Reader() { initialize_callback(cb); }

  virtual void decode(pb_istream_t *stream) override {
    if (!pb_decode(stream, onnx_ModelProto_fields, &proto)) {
      throw std::runtime_error("Decoding failed");
    }

    value.graph = std::move(graph_reader.value);
  }

  static Model_ read(pb_istream_t *stream) {
    Reader<Model_> reader;
    reader.decode(stream);
    return reader.value;
  }

  onnx_ModelProto proto;
  Reader<Graph_> graph_reader;
  Model_ value;
};


// IR graph construction

at::Tensor buildTensor(const Tensor_& tensor_) {

  at::Tensor tensor;

  switch(tensor_.data_type) {
    case onnx_TensorProto_DataType_UINT8:
      tensor = at::CPU(at::kByte).tensor();
      break;
    case onnx_TensorProto_DataType_INT8:
      tensor = at::CPU(at::kChar).tensor();
      break;
    case onnx_TensorProto_DataType_INT16:
      tensor = at::CPU(at::kShort).tensor();
      break;
    case onnx_TensorProto_DataType_INT32:
      tensor = at::CPU(at::kInt).tensor();
      break;
    case onnx_TensorProto_DataType_INT64:
      tensor = at::CPU(at::kLong).tensor();
      break;
    case onnx_TensorProto_DataType_FLOAT16:
      tensor = at::CPU(at::kHalf).tensor();
      break;
    case onnx_TensorProto_DataType_FLOAT:
      tensor = at::CPU(at::kFloat).tensor();
      break;
    case onnx_TensorProto_DataType_DOUBLE:
      tensor = at::CPU(at::kDouble).tensor();
      break;
    default:
      throw std::runtime_error("Unsupported data type");
  }

  tensor.resize_(tensor_.dims);

  TORCH_ASSERT(tensor.storage()->size() * tensor.storage()->elementSize() == tensor_.raw_data.size());

  std::memcpy(tensor.data_ptr(), tensor_.raw_data.data(), tensor_.raw_data.size());

  return tensor;
}

Graph_ readSubgraph(const std::string& serialized_subgraph) {
  pb_istream_t istream = pb_istream_from_buffer(reinterpret_cast<const pb_byte_t *>(serialized_subgraph.data()), serialized_subgraph.size());

  return Reader<Graph_>::read(&istream);
}

void buildBlock(const Graph_& graph_, Block* block,
                std::unordered_map<std::string, Value*>& value_map);

void buildBlocks(const std::vector<Graph_>& graphs_, Node* node,
                 std::unordered_map<std::string, Value*>& value_map) {
  for (auto g_ : graphs_) {
    auto block = node->addBlock();
    buildBlock(g_, block, value_map);
  }
}

std::shared_ptr<Graph> buildGraph(const Graph_& graph_) {
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> value_map;

  buildBlock(graph_, graph->block(), value_map);

  return graph;
}

void buildBlock(const Graph_& graph_, Block* block,
                std::unordered_map<std::string, Value*>& value_map) {

  for (auto & input : graph_.inputs) {
    value_map[input.name] = block->addInput();
  }

  for (auto & node_ : graph_.nodes) {
    TORCH_ASSERT(node_.op_type != "CppOp");
    TORCH_ASSERT(node_.op_type != "PythonOp");

    auto node = block->owningGraph()->create(Symbol::fromDomainAndUnqualString(node_.domain, node_.op_type),
                                             node_.outputs.size());

    for (auto & attr : node_.attrs) {
      Symbol name = Symbol::attr(attr.name);

      switch(attr.type) {
        case onnx_AttributeProto_AttributeType_UNDEFINED:
          throw std::runtime_error("UNDEFINED attribute unsupported");
          break;
        case onnx_AttributeProto_AttributeType_FLOAT:
          node->f_(name, attr.f);
          break;
        case onnx_AttributeProto_AttributeType_INT:
          node->i_(name, attr.i);
          break;
        case onnx_AttributeProto_AttributeType_STRING:
          node->s_(name, std::move(attr.s));
          break;
        case onnx_AttributeProto_AttributeType_TENSOR:
          node->t_(name, buildTensor(attr.t));
          break;
        case onnx_AttributeProto_AttributeType_GRAPH:
          node->g_(name, buildGraph(readSubgraph(attr.g)));
          break;
        case onnx_AttributeProto_AttributeType_FLOATS:
          node->fs_(name, std::move(attr.fs));
          break;
        case onnx_AttributeProto_AttributeType_INTS:
          node->is_(name, std::move(attr.is));
          break;
        case onnx_AttributeProto_AttributeType_STRINGS:
          node->ss_(name, std::move(attr.ss));
          break;
        case onnx_AttributeProto_AttributeType_TENSORS:
          node->ts_(name, fmap(attr.ts, [](const Tensor_& t) { return buildTensor(t); }));
          break;
        case onnx_AttributeProto_AttributeType_GRAPHS:
          if (attr.name == "_blocks") {
            buildBlocks(fmap(attr.gs, [](const std::string& g) { return readSubgraph(g); }), node, value_map);
          }
          else {
            node->gs_(name, fmap(fmap(attr.gs, [](const std::string& g) { return readSubgraph(g); } ),
                                               [](const Graph_& g_) { return buildGraph(g_); }));
          }
          break;
      }
    }

    for (auto & input : node_.inputs) {
      auto v = value_map[input];
      node->addInput(v);
    }

    for (size_t i=0; i<node_.outputs.size(); i++) {
      value_map[node_.outputs[i]] = node->outputs()[i];
    }

    block->appendNode(node);
  }

  for (auto & output : graph_.outputs) {
    Value* v = value_map.at(output.name);
    block->registerOutput(v);
  }
}

std::shared_ptr<Graph> buildGraph(const Graph_& graph_, std::vector<at::Tensor>& initializers) {

  auto graph = buildGraph(graph_);

  for (auto tensor_ : graph_.initializers) {
    initializers.push_back(buildTensor(tensor_));
  }

  return graph;
}

}

std::shared_ptr<Graph> ImportIRGraph(const std::string& serialized_graph,
                                     std::vector<at::Tensor>& initializers) {

  pb_istream_t istream = pb_istream_from_buffer(reinterpret_cast<const pb_byte_t *>(serialized_graph.data()), serialized_graph.size());

  auto model = Reader<Model_>::read(&istream);

  auto graph = buildGraph(model.graph, initializers);

  return graph;
}

}}
