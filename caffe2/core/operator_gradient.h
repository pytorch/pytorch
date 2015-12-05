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


struct GetGradientDefBaseVerbose {
 public:
  GetGradientDefBaseVerbose(
      const bool copy_device_option, const bool copy_engine,
      const bool copy_args)
      : copy_device_option_(copy_device_option), copy_engine_(copy_engine),
      copy_args_(copy_args) {}
  virtual ~GetGradientDefBaseVerbose() {}

  bool CopyDeviceOption() const { return copy_device_option_; }
  bool CopyEngine() const { return copy_engine_; }
  bool CopyArguments() const { return copy_args_; }
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

  virtual vector<OperatorDef>* Create(const OperatorDef& def) {
    NOT_IMPLEMENTED;
    return nullptr;
  }

  template <class... Args>
  inline static vector<OperatorDef>* SingleGradientDef(Args ... args) {
    return new vector<OperatorDef>{CreateOperatorDef(args...)};
  }

  bool copy_device_option_;
  bool copy_engine_;
  bool copy_args_;
};


struct GetGradientDefBase : public GetGradientDefBaseVerbose {
 public:
  GetGradientDefBase() : GetGradientDefBaseVerbose(true, true, true) {}
};

struct NoGradient : public GetGradientDefBase {
  vector<OperatorDef>* Create(const OperatorDef& def) override {
    return new vector<OperatorDef>();
  }
};

// This is used when the operator definition is designed to not have a gradient.
// Calling a gradient on this operator def will cause Caffe2 to throw the towel.
struct ThrowTheTowelIfGradientIsCalled : public GetGradientDefBase {
  vector<OperatorDef>* Create(const OperatorDef& def) override;
};

// This should only be used sparsely when the gradient does exist, but we have
// not implemented it yet.
struct GradientNotImplementedYet : public GetGradientDefBase {
  vector<OperatorDef>* Create(const OperatorDef& def) override;
};

vector<OperatorDef>* CreateGradientDefsInternal(
    const OperatorDef& def, GetGradientDefBaseVerbose* obj);

template <class GetGradientDef>
class GradientRegisterer {
 public:
  GradientRegisterer(const string& key) {
    GradientRegistry()->Register(
        key, GradientRegisterer<GetGradientDef>::Creator);
  }

  static vector<OperatorDef>* Creator(const OperatorDef& def) {
    return CreateGradientDefsInternal(def, new GetGradientDef());
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

#define GRADIENT_NOT_IMPLEMENTED_YET(name)                                     \
  GradientRegisterer<GradientNotImplementedYet>                                \
      g_GradientRegisterer_##name(#name)

// Creates the gradient operators of a given operator definition.
inline vector<OperatorDef>* GetGradientDefs(const OperatorDef& def) {
  return GradientRegistry()->Create(def.type(), def);
}

}  // namespace caffe2

#endif  // CAFFE2_CORE_OPERATOR_GRADIENT_H_
