#include <ATen/core/opschema/layer_norm.h>
#include <ATen/core/dispatch/OpSchemaRegistration.h>

namespace c10 {
namespace core {
namespace opschema {
  // TODO Parse schema string instead of creating FunctionSchema manually
  C10_DEFINE_OP_SCHEMA(LayerNorm, FunctionSchema(
      "caffe2::layer_norm_dont_use_this_op_yet",
      (std::vector<c10::Argument>{
        c10::Argument("input"),
        c10::Argument("axis", IntType::get()),
        c10::Argument("epsilon", FloatType::get()),
        c10::Argument("output", OptionalType::ofTensor(), c10::nullopt, IValue()),
        c10::Argument("output_mean", OptionalType::ofTensor(), c10::nullopt, IValue()),
        c10::Argument("output_stdev", OptionalType::ofTensor(), c10::nullopt, IValue())
      }), (std::vector<c10::Argument>{
        c10::Argument("output"),
        c10::Argument("mean"),
        c10::Argument("stdev")
      })
  ));
}
}
}
