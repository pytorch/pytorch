#ifndef CAFFE2_CORE_OPERATOR_GRADIENT_H_
#define CAFFE2_CORE_OPERATOR_GRADIENT_H_

#include "caffe2/core/registry.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {

// Utility functions for gradient computation.
inline string GradientName(const string& name) {
  return name + "_grad";
}

DECLARE_REGISTRY(GradientRegistry, vector<OperatorDef>, const OperatorDef&);

template <class GetGradientDef>
class GradientRegisterer {
 public:
  GradientRegisterer(const string& key) {
    GradientRegistry()->Register(
        key, GradientRegisterer<GetGradientDef>::Creator);
  }

  static vector<OperatorDef>* Creator(const OperatorDef& def) {
    CAFFE_VLOG(1) << "Creator: " << def.DebugString();
    vector<OperatorDef>* grad_defs = GetGradientDef::Create(def);
    CAFFE_CHECK(grad_defs != nullptr);
    // Copy device option if needed.
    if (GetGradientDef().CopyDeviceOption() && def.has_device_option()) {
      for (OperatorDef& grad_def : *grad_defs) {
        grad_def.mutable_device_option()->CopyFrom(def.device_option());
      }
    }
    // Copy engine if needed.
    if (GetGradientDef().CopyEngine() && def.has_engine()) {
      for (OperatorDef& grad_def : *grad_defs) {
        grad_def.set_engine(def.engine());
      }
    }
    // Copy arguments if needed.
    if (GetGradientDef().CopyArguments() && def.arg_size()) {
      for (OperatorDef& grad_def : *grad_defs) {
        grad_def.mutable_arg()->CopyFrom(def.arg());
      }
    }
    for (const OperatorDef& grad_def : *grad_defs) {
      CAFFE_VLOG(1) << "Gradient: " << grad_def.DebugString();
    }
    return grad_defs;
  }
};

template <bool copy_device_option, bool copy_engine, bool copy_args>
struct GetGradientDefBaseVerbose {
  constexpr bool CopyDeviceOption() const { return copy_device_option; }
  constexpr bool CopyEngine() const { return copy_engine; }
  constexpr bool CopyArguments() const { return copy_args; }
  inline static string I(const OperatorDef& def, const int i) {
    return def.input(i);
  }
  inline static string O(const OperatorDef& def, const int i) {
    return def.output(i);
  }
  inline static string GI(const OperatorDef& def, const int i) {
    return GradientName(def.input(i));
  }
  inline static string GO(const OperatorDef& def, const int i) {
    return GradientName(def.output(i));
  }
  template <class... Args>
  inline static vector<OperatorDef>* SingleGradientDef(Args ... args) {
    return new vector<OperatorDef>{CreateOperatorDef(args...)};
  }
};
typedef struct GetGradientDefBaseVerbose<true, true, true> GetGradientDefBase;

struct NoGradient : public GetGradientDefBase {
  static vector<OperatorDef>* Create(const OperatorDef& def) {
    return new vector<OperatorDef>();
  }
};

struct ThrowTheTowelIfGradientIsCalled : public GetGradientDefBase {
  static vector<OperatorDef>* Create(const OperatorDef& def) {
    CAFFE_LOG_FATAL << "You should not call the gradient of operator of type "
                    << def.type();
    // Just to suppress compiler warnings
    return new vector<OperatorDef>();
  }
};

struct GradientNotImplementedYet : public GetGradientDefBase {
  static vector<OperatorDef>* Create(const OperatorDef& def) {
    CAFFE_LOG_FATAL << "Gradient for operator type "
                    << def.type() << " has not been implemented yet.";
  }
};

#define REGISTER_GRADIENT(name, GetGradientDef)                                \
  GradientRegisterer<GetGradientDef> g_GradientRegisterer_##name(#name)

// NO_GRADIENT means that the operator does not need any gradient computation.
#define NO_GRADIENT(name)                                                      \
  GradientRegisterer<NoGradient> g_GradientRegisterer_##name(#name)

// SHOULD_NOT_DO_GRADIENT means that the operator is not designed to have
// gradient operators. If you attempt to call the gradient, a log fatal will
// occur.
#define SHOULD_NOT_DO_GRADIENT(name)                                           \
  GradientRegisterer<ThrowTheTowelIfGradientIsCalled>                          \
      g_GradientRegisterer_##name(#name)

// SHOULD_NOT_DO_GRADIENT means that the operator is not designed to have
// gradient operators. If you attempt to call the gradient, a log fatal will
// occur.
#define SHOULD_NOT_DO_GRADIENT(name)                                           \
  GradientRegisterer<ThrowTheTowelIfGradientIsCalled>                          \
      g_GradientRegisterer_##name(#name)

#define GRADIENT_NOT_IMPLEMENTED_YET(name)                                     \
  GradientRegisterer<GradientNotImplementedYet>                                \
      g_GradientRegisterer_##name(#name)

// Creates the gradient operators of a given operator definition.
inline vector<OperatorDef>* GetGradientDefs(const OperatorDef& def) {
  return GradientRegistry()->Create(def.type(), def);
}

}  // namespace caffe2

#endif  // CAFFE2_CORE_OPERATOR_GRADIENT_H_
