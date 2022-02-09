//#include <test/cpp/jit/test_utils.h>

#include <gtest/gtest.h>
#include <c10/core/impl/SizesAndStrides.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Registry.h>

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}



// Tests go in torch::jit
namespace torch {
namespace embedded {

// Tensor should be simple, but usable in operators
class Tensor {
  c10::ScalarType type;
  void* data;
  c10::impl::SizesAndStrides sizes_strides; // may be simplified later
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

// IValue is used to unify input/output types in a Node
// It's possible that an op argument is any of the types.
// in native_functions.yaml, types are Tensor, Scalar, int[], float[], bool,
// tag + union of payload
// Use union of prim or pointers to simplify the structure and storage
struct IValue {
  Tag tag;
  union Payload {
    int64_t as_int;
    double as_double;
    bool as_bool;
    Tensor* as_tensor;
    IntList* as_int_list;
    DoubleList* as_double_list;
    BoolList* as_bool_list;
  };
  Payload payload;

  IValue(int64_t i) : tag(Tag::Int) {
    payload.as_int = i;
  }

  bool isInt() {
    return tag == Tag::Int;
  }

  int toInt() {
    AT_ASSERT(isInt());
    return payload.as_int;
  }
};

struct IValueList {
  int size;
  IValue values[];
};

// operator registry

struct Node {
  IValueList* inputs;
  IValueList* outputs;
};

class Interpreter {

};

class Graph {
  IValueList* inputs;
  IValueList* outputs;
  IValueList* intermediates;

};

// Operators
using Operator = std::function<void(Node*)>;
C10_DECLARE_REGISTRY(OperatorRegistry, Operator);

void Op_test(Node* node) {

}
//C10_REGISTER_CLASS(OperatorRegistry, name, OperatorFunctor);


TEST(EmbeddedTest, Simple) {
  IValue a(1), b(2);
  IValue out(0);
  Node node;
}
} // namespace embedded
} // namespace torch
