#ifndef CAFFE2_OPERATORS_NO_DEFAULT_ENGINE_OP_H_
#define CAFFE2_OPERATORS_NO_DEFAULT_ENGINE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

/**
 * A helper class to denote that an op does not have a default engine.
 *
 * NoDefaultEngineOp is a helper class that one can use to denote that a
 * specific operator is not intended to be called without an explicit engine
 * given. This is the case for e.g. the communication operators where one has
 * to specify a backend (like MPI or ZEROMQ).
 */
template <class Context>
class NoDefaultEngineOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(NoDefaultEngineOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    CAFFE_THROW(
        "The operator ",
        this->debug_def().type(),
        " does not have a default engine implementation. Please "
        "specify an engine explicitly for this operator.");
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_NO_DEFAULT_ENGINE_OP_H_
