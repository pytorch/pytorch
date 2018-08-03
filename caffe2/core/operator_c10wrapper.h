#pragma once

#include "caffe2/core/dispatch/Dispatcher.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

/**
 * To make a c10 operator "C10Add" callable from caffe2 as "C2MyAddOpName", just
 * write
 *
 *     REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH(C10Add, C2MyAddOpName)
 *
 * Note: This wrapper currently only supports C10 ops that have exactly one
 * output and take that in the last parameter as "Tensor* output".
 * TODO: Figure out a better way to handle output parameters
 */

template <class OpSchemaDef, class Context>
class C10OperatorWrapper final : public Operator<Context> {
  using Schema = c10::OpSchema<OpSchemaDef>;

 public:
  C10OperatorWrapper(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    RunOnDevice_(
        c10::guts::make_index_sequence<Schema::signature::num_args - 1>());
    return true;
  }

 private:
  template <size_t... InputIndex>
  void RunOnDevice_(c10::guts::index_sequence<InputIndex...>) {
    c10::Dispatcher<OpSchemaDef>::call(Input(InputIndex)..., Output(0));
  }
};

CAFFE_DECLARE_REGISTRY(
    C10OperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);

// TODO Currently we only register the CPU variant. This is going to be fixed
//      once the tensor detemplatization lands.
#define REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH(OpSchemaDef, Name) \
  CAFFE_REGISTER_CLASS(                                              \
      C10OperatorRegistry, Name, C10OperatorWrapper<OpSchemaDef, CPUContext>)

} // namespace caffe2
