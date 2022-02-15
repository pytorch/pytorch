//#include <test/cpp/jit/test_utils.h>

#include <gtest/gtest.h>
#include <c10/core/impl/SizesAndStrides.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Registry.h>
// HACK: use existing ff schema before the final embedded schema is set.
#include <test/cpp/embedded/schema_generated.h>

#ifndef ACTIVATION_BUF_SZ
#warning  "ACTIVATION_BUF_SZ needs to be defined. Using default value"
#define ACTIVATION_BUF_SZ       0x00200000
#endif  /* ACTIVATION_BUF_SZ */

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}



// Tests go in torch::jit
namespace torch {
namespace embedded {

void error_with_message(char* message) {
  // A hacky error function before we have a good convention,
  // better without exception.
  printf("%s\n", message);
  exit(EXIT_FAILURE);
}

// Memory buffers
// 1. Model buffer in flatbuffer format.
// TODO: codegen the content and put it in a seperate file.
// TODO: define __attribute__ of sections
static const uint8_t program[] = {};

// Scratch space to store R/W activation tensors.
// If the size is known AOT it's' set at compile time (here).
// Replace it with malloc when AOT size is not available.
// Usually the memory budget is set AOT, so a max size if required
// at compile time. Runtime error is thrown when it goes beyond the budget.
// TODO: define __attribute__ of sections
// TODO: allow multiple buffers
static uint8_t activation_buffer[ACTIVATION_BUF_SZ];

struct IntList {
  int size;
  int data[];
};

struct DoubleList {
  int size;
  double data[];
};

struct BoolList {
  int size;
  bool data[];
};

// Tensor should be simple, but usable in operators
// TODO: APIs common to at::Tensor
struct Tensor {
  c10::ScalarType type;
  void* data = nullptr;
  int dim;
  int* sizes;
  // Strides
  // Quantizer
};

#define TORCH_FORALL_TAGS(_) \
  _(None)                    \
  _(Tensor)                  \
  _(Double)                  \
  _(Int)                     \
  _(Bool)                    \
  _(ListDouble)                  \
  _(ListInt)                     \
  _(ListBool)

enum class Tag : uint32_t {
#define DEFINE_TAG(x) x,
  TORCH_FORALL_TAGS(DEFINE_TAG)
#undef DEFINE_TAG
};

// Value is used to unify input/output types in a Node
// It's possible that an op argument is any of the types.
// in native_functions.yaml, types are Tensor, Scalar, int[], float[], bool,
// tag + union of payload
// Use union of prim or pointers to simplify the structure and storage
struct Value {
  Tag tag;
  union Payload {
    int64_t as_int;
    double as_double;
    bool as_bool;
    // Raw pointers for now instead of intrusive_ptr, because some embedded
    // systems may not support atomic ref count.
    Tensor* as_tensor;
    IntList* as_int_list;
    DoubleList* as_double_list;
    BoolList* as_bool_list;
  };
  Payload payload;

  Value(int64_t i) : tag(Tag::Int) {
    payload.as_int = i;
  }

  bool isInt() const {
    return tag == Tag::Int;
  }

  int toInt() const {
    if (!isInt()) {
      error_with_message("Value is not an int.");
    }
    return payload.as_int;
  }

  Value(Tensor* t) : tag(Tag::Tensor) {
    payload.as_tensor = t;
  }

  bool isTensor() const {
    return tag == Tag::Tensor;
  }

  Tensor* toTensor() const {
    if (!isTensor()) {
      error_with_message("Value is not a tensor.");
    }
    return payload.as_tensor;
  }
};

struct ValueList {
  int size;
  Value values[];
};

// operator registry

//// Later the args may be extended to a node with args and other info associated.
//struct Node {
//  ValueList* inputs;
//  ValueList* outputs;
//};

class MemoryManager {

};

class Interpreter {
  serialization::Program module;
  MemoryManager memoryManager;

  int Init() {
    // Allocate scratch buffers
    // address mapping
    return 0;
  }
  //
};

class Graph {
  ValueList* inputs;
  ValueList* outputs;
  ValueList* intermediates;

};


//// Operators
//using Operator = std::function<void(ValueList*)>;
//C10_DECLARE_REGISTRY(OperatorRegistry, Operator);
//
//void Op_add_int(ValueList* args) {
//  // Unboxing, can be code-gen. and the kernel.
//  args->values[2].payload.as_int = args->values[0].toInt() + args->values[1].toInt();
//}

//C10_REGISTER_CLASS(OperatorRegistry, ATen::Add.int, Op_add_int);
struct serializer {


flatbuffers::Offset<serialization::Tensor> tensorToFB(
    flatbuffers::FlatBufferBuilder& fbb,
    const Value& ivalue) {
  auto tensor = ivalue.toTensor();

  std::vector<int> sizes{tensor->sizes[0], tensor->sizes[0] + tensor->dim};

  return serialization::CreateTensorDirect(
      fbb,
      storage_index_++,
      static_cast<int8_t>(tensor->type),
      0, // hard code storage offset for now
      &sizes,
      false, // requires grad
      0);
}

flatbuffers::Offset<serialization::Value> valueToFB(flatbuffers::FlatBufferBuilder& fbb, const Value& ivalue) {
  flatbuffers::Offset<void> offset = 0;
  serialization::ValueUnion ivalue_type = serialization::ValueUnion::NONE;
  if (ivalue.tag == Tag::Tensor) {

    ivalue_type = serialization::ValueUnion::Tensor;
    offset = tensorToFB(fbb, ivalue).Union();
  } else {
    // TODO: Support other types.
    error_with_message("Type not supported yet.");
  }
}

uint32_t storeValueAndGetIndex(
    flatbuffers::FlatBufferBuilder& fbb,
    const Value& ivalue) {
  auto offset = valueToFB(fbb, ivalue);
  uint32_t index = insertIValue(offset);
  return index;
}

int storage_index_;
};

TEST(EmbeddedTest, Simple) {

  // Prepare for the flatbuffer program
  flatbuffers::FlatBufferBuilder fbb;
  serialization::ProgramBuilder pb(fbb);
  pb.add_version(1);

  // Prepare for a graph
  // It has two operations, mul and add to finish
  // z = a * x
  // y = z * b
  // TODO: Add other types like IntList or BoolList
  serialization::GraphBuilder gb(fbb);
  // Values: a, b, x, y and intermediate z (ax), all tensors
  // Constant tensors a and b have data.
  std::vector<Value> values; // TODO: use values;

  Tensor a;
  a.type = c10::ScalarType::Int;
  a.dim = 2;
  a.sizes = new int[a.dim]{2, 2};
  int a_data[4]{1, 1, 1, 1};
  a.data = a_data;
  values.emplace_back(&a);

  Tensor b;
  b.type = c10::ScalarType::Int;
  b.dim = 2;
  b.sizes = new int[b.dim]{2, 2};
  int b_data[4]{2, 2, 2, 2};
  b.data = b_data;
  values.emplace_back(&b);

  // Rest of tensors (x, z, y) don't have data
  Tensor x;
  x.type = c10::ScalarType::Int;
  x.dim = 2;
  x.sizes = new int[x.dim]{2, 2};

  Tensor y;
  y.type = c10::ScalarType::Int;
  y.dim = 2;
  y.sizes = new int[y.dim]{2, 2};

  Tensor z;
  z.type = c10::ScalarType::Int;
  z.dim = 2;
  z.sizes = new int[z.dim]{2, 2};

  values.emplace_back(&x);
  values.emplace_back(&y);
  values.emplace_back(&z);

  for (const auto& v : values) {
    auto index = storeValueAndGetIndex(fbb, v);
  }
  auto value_type = serialization::ValueUnion::Int;
  auto offset = fbb.CreateStruct(serialization::Int(2)).Union();

//  flatbuffers::Vector<

  auto p = pb.Finish();
  fbb.Finish(p);
  uint8_t* buff = fbb.GetBufferPointer();
  auto program = serialization::GetProgram(buff);


//  node.inputs = new ValueList
}
} // namespace embedded
} // namespace torch
