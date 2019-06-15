// WARNING! WARNING! WARNING!
// This file is a temporary hack to enable development of pytorch quantization
//
// It effectively wraps Caffe2 ops as is through custom jit ops mechanism
// It obviously has terrible performance - caffe2 operator instance is created
// on each invocation and also creation involves creating a protobuf (sigh...)
//
// Our plan is to implement quantized operators natively in c10 as operators and
// also enforce some additional contracts on operator semantics:
// - explicitly express weights prepacking as a separate operator to signify
//   reliance on weights being constant
// - don't modify arguments of the op (OperatorDef) to store data
// - explicitly model figuring out quantization params for dynamic quantization
//   instead of memorizing the first batch's params

#include <torch/csrc/jit/custom_operator.h>
#include <torch/csrc/jit/operator.h>

#include <caffe2/core/operator.h>
#include <caffe2/core/tensor_int8.h>
#include <torch/csrc/autograd/variable.h>

namespace torch {
namespace jit {

using caffe2::int8::Int8TensorCPU;

namespace {

caffe2::Tensor from_at_tensor(const c10::IValue& v) {
  return caffe2::Tensor(autograd::Variable(std::move(v).toTensor()).tensor_data());
}

Int8TensorCPU from_proxy(const c10::IValue& proxy) {
  auto t = proxy.toTuple()->elements();
  Int8TensorCPU r;
  r.t = from_at_tensor(t[0]);
  r.scale = t[1].toDouble();
  r.zero_point = t[2].toInt();
  return r;
}

at::Tensor to_proxy(const caffe2::Tensor& t) {
  return autograd::make_variable(at::Tensor(t.UnsafeSharedInstance()), false);
}

c10::intrusive_ptr<c10::ivalue::Tuple> to_proxy(const Int8TensorCPU& t) {
  return c10::ivalue::Tuple::create({to_proxy(t.t), t.scale, t.zero_point});
}

// TODO: replace this with c10 registration when it's ready
RegisterOperators reg({
    Operator(
        // NOTE: we put outout in double parens because it's an output of type
        // tuple, not a tuple of multiple outputs
        "c10::quantized_relu((Tensor, float, int) self) -> ((Tensor, float, int))",
        // TODO: can't use C++ inference - doesn't work yet for tuple types
        [](Stack& stack) {
          AT_ASSERT(caffe2::GetRegisteredOperators().count(
              caffe2::OpRegistryKey("Relu", "DNNLOWP")))

          // TODO: refactor the underlying op implementation and inline it in
          // c10 kernel
          caffe2::Workspace ws;
          ws.CreateBlob("X")->Reset(
              new Int8TensorCPU(from_proxy(std::move(peek(stack, 0, 1)))));

          auto def = caffe2::CreateOperatorDef(
              "Relu", "proxy", {"X"}, {"Y"}, caffe2::DeviceOption(), "DNNLOWP");
          auto op = caffe2::CreateOperator(def, &ws);

          op->Run();

          drop(stack, 1);
          pack(stack, to_proxy(ws.GetBlob("Y")->Get<Int8TensorCPU>()));
          return 0;
        }),

    Operator(
        "c10::quantize(Tensor X, float? scale = None, int? zero_point = None) -> ((Tensor, float, int))",
        [](Stack& stack) {
          AT_ASSERT(caffe2::GetRegisteredOperators().count(
              caffe2::OpRegistryKey("Quantize", "DNNLOWP")))

          // TODO: refactor the underlying op implementation and inline it in
          // c10 kernel
          caffe2::Workspace ws;
          ws.CreateBlob("X")->Reset(
              new caffe2::Tensor(from_at_tensor(std::move(peek(stack, 0, 3)))));

          auto def = caffe2::CreateOperatorDef(
              "Quantize",
              "proxy",
              {"X"},
              {"Y"},
              caffe2::DeviceOption(),
              "DNNLOWP");
          auto s = peek(stack, 1, 3).toOptional<float>();
          if (s.has_value()) {
            def.add_arg()->CopyFrom(caffe2::MakeArgument("Y_scale", *s));
          }
          auto zp = peek(stack, 2, 3).toOptional<int32_t>();
          if (zp.has_value()) {
            def.add_arg()->CopyFrom(caffe2::MakeArgument("Y_zero_point", *zp));
          }
          auto op = caffe2::CreateOperator(def, &ws);

          op->Run();

          drop(stack, 3);
          pack(stack, to_proxy(ws.GetBlob("Y")->Get<Int8TensorCPU>()));
          return 0;
        }),

    Operator(
        "c10::dequantize((Tensor, float, int) x_q) -> Tensor",
        // TODO: can't use C++ inference - doesn't work yet for tuple types
        [](Stack& stack) {
          AT_ASSERT(caffe2::GetRegisteredOperators().count(
              caffe2::OpRegistryKey("Dequantize", "DNNLOWP")))

          // TODO: refactor the underlying op implementation and inline it in
          // c10 kernel
          caffe2::Workspace ws;
          ws.CreateBlob("X")->Reset(
              new Int8TensorCPU(from_proxy(std::move(peek(stack, 0, 1)))));

          auto def = caffe2::CreateOperatorDef(
              "Dequantize",
              "proxy",
              {"X"},
              {"Y"},
              caffe2::DeviceOption(),
              "DNNLOWP");
          auto op = caffe2::CreateOperator(def, &ws);

          op->Run();

          drop(stack, 1);
          pack(stack, to_proxy(ws.GetBlob("Y")->Get<caffe2::Tensor>()));
          return 0;
        }),
});
} // namespace
} // namespace jit
} // namespace torch
