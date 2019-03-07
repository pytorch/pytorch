#ifndef CAFFE2_CORE_OPERATOR_GRADIENT_H_
#define CAFFE2_CORE_OPERATOR_GRADIENT_H_

#include "c10/util/Registry.h"
#include "caffe2/core/operator_schema.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

/* @brief A struct that abstracts on top of dense and sparse blobs.
 *
 * For a dense blob, its gradient name should be written into dense_, and for
 * a sparse blob, its gradient name should be written into indice_ for
 * the sparse indices and value_ for the values.
 */
struct CAFFE2_API GradientWrapper {
  string dense_;
  string indices_;
  string values_;

  inline bool IsDense() const {
    return (dense_.size() != 0);
  }
  inline bool IsSparse() const {
    return (indices_.size() != 0 || values_.size() != 0);
  }
  inline bool IsEmpty() const {
    return (!IsDense() && !IsSparse());
  }
};

/**
 * A struct that holds the gradient operators and related gradient maps.
 */
struct CAFFE2_API GradientOpsMeta {
  vector<OperatorDef> ops_;
  vector<GradientWrapper> g_input_;

  GradientOpsMeta() {}
  GradientOpsMeta(
      const vector<OperatorDef>& ops,
      const vector<GradientWrapper>& v)
      : ops_(ops), g_input_(v) {}
};

class CAFFE2_API GradientMakerBase {
 public:
  GradientMakerBase(
      const OperatorDef& def,
      const vector<GradientWrapper>& g_output)
      : def_(def), g_output_(g_output), g_input_(def.input_size()){};
  virtual ~GradientMakerBase() {}
  virtual bool CopyDeviceOption() const {
    return true;
  }
  virtual bool CopyEngine() const {
    return true;
  }
  virtual bool CopyArguments() const {
    return true;
  }

  virtual void VerifyOp() const {
    auto* schema = OpSchemaRegistry::Schema(def_.type());
    if (schema) {
      CAFFE_ENFORCE(
          schema->Verify(def_),
          "(GradientMaker) Operator def did not pass schema checking: ",
          ProtoDebugString(def_));
    }
  }

  /**
   * @brief Returns the gradient ops meta.
   *
   * If your gradient op generator only use standard input and output
   * manipulations, you can simply implement GetGradientDefs() that
   * returns vector<OperatorDef>. In that, you can call GI, GI_V and GI_I
   * that will automatically create the gradient registration for you.
   *
   * If you need to do custom gradient name registration, overload this
   * function directly.
   */
  virtual GradientOpsMeta Get() {
    VerifyOp();
    vector<OperatorDef> new_defs = GetGradientDefs();
    for (auto& opdef : new_defs) {
      opdef.set_is_gradient_op(true);
    }
    return GradientOpsMeta(new_defs, g_input_);
  };

  const OperatorDef& Def() const {
    return def_;
  }

 protected:
  virtual vector<OperatorDef> GetGradientDefs() {
    CAFFE_NOT_IMPLEMENTED;
  }

  // Helper functions to return names for the gradient computation.
  // I(idx), O(idx): return the input and output names.
  // GO(idx): return the name of the gradient for output idx.
  // GI(idx), GI_I(idx), GI_V(idx): return the name of the gradient for
  //     input idx, and also registers that name into the gradient
  //     registry to be returned.
  string I(const int i) {
    CAFFE_ENFORCE((i >= 0) && (i < def_.input().size()));
    return def_.input(i);
  }
  string O(const int i) {
    CAFFE_ENFORCE((i >= 0) && (i < def_.output().size()));
    return def_.output(i);
  }
  string GI(const int i) {
    CAFFE_ENFORCE(
        !g_input_.at(i).IsSparse(),
        "Input ",
        def_.input(i),
        " already set to sparse.");
    g_input_.at(i).dense_ = GradientName(def_.input(i));
    return GradientName(def_.input(i));
  }
  string GI_I(const int i) {
    CAFFE_ENFORCE(
        !g_input_.at(i).IsDense(),
        "Input ",
        def_.input(i),
        " already set to dense.");
    g_input_.at(i).indices_ = GradientSliceIndices(def_.input(i));
    return GradientSliceIndices(def_.input(i));
  }
  string GI_V(const int i) {
    CAFFE_ENFORCE(
        !g_input_.at(i).IsDense(),
        "Input ",
        def_.input(i),
        " already set to dense.");
    g_input_.at(i).values_ = GradientSliceValues(def_.input(i));
    return GradientSliceValues(def_.input(i));
  }
  string GO(const int i) {
    CAFFE_ENFORCE(
        g_output_.at(i).IsDense(),
        "Gradient of output ",
        def_.output(i),
        (g_output_.at(i).IsSparse() ? " is sparse (expected dense)."
                                    : " is not provided!"));
    return g_output_.at(i).dense_;
  }
  string GO_I(const int i) {
    CAFFE_ENFORCE(
        g_output_.at(i).IsSparse(),
        "Gradient of output ",
        def_.output(i),
        (g_output_.at(i).IsDense() ? " is dense (expected sparse)."
                                   : " is not provided!"));
    return g_output_.at(i).indices_;
  }
  string GO_V(const int i) {
    CAFFE_ENFORCE(
        g_output_.at(i).IsSparse(),
        "Gradient of output ",
        def_.output(i),
        (g_output_.at(i).IsDense() ? " is dense (expected sparse)."
                                   : " is not provided!"));
    return g_output_.at(i).values_;
  }
  const GradientWrapper& GradOut(int i) {
    return g_output_.at(i);
  }

  // Function to add a gradient pair to map.
  void SetDense(const int i, const string& name) {
    CAFFE_ENFORCE(
        !g_input_.at(i).IsSparse(),
        "Input ",
        def_.input(i),
        " already set to sparse.");
    g_input_.at(i).dense_ = name;
  }
  void SetSparse(const int i, const string& indices, const string& values) {
    CAFFE_ENFORCE(
        !g_input_.at(i).IsDense(),
        "Input ",
        def_.input(i),
        " already set to dense.");
    g_input_.at(i).indices_ = indices;
    g_input_.at(i).values_ = values;
  }

  /**
   * @brief a helper function to allow one to create one single operator
   * def, which is usually the case for many simple operators.
   */
  template <class... Args>
  inline static vector<OperatorDef> SingleGradientDef(const Args&... args) {
    return vector<OperatorDef>{CreateOperatorDef(args...)};
  }

 public:
  /**
    * Returns map that returns the parameters that the gradients are for.
    */
  static CaffeMap<string, string> MatchGradsToParams(const OperatorDef& op) {
    // NOTE: how to go beyond string-matching?
    CaffeMap<string, string> m;
    for (auto& out : op.output()) {
      if (IsGradientBlob(out)) {
        m[out] = out.substr(0, out.length() - 5);
      }
    }
    return m;
  }

 private:
  // Utility functions for gradient name computation. We don't expose them
  // in order to discourage the use of such names explicitly.
  static string GradientName(const string& name) {
    return name + "_grad";
  }

  static bool IsGradientBlob(const string& name) {
    return name.length() > 5 && name.find("_grad") == name.length() - 5;
  }

  static string GradientNameToParam(const string& name) {
    CHECK(IsGradientBlob(name));
    return name.substr(0, name.length() - 5);
  }

  static string GradientSliceIndices(const string& name) {
    return name + "_grad_indices";
  }

  static string GradientSliceValues(const string& name) {
    return name + "_grad_values";
  }

 protected:
  // We make the member variables protected in case someone wants to write
  // a fully custom Get() function.
  const OperatorDef& def_;
  const vector<GradientWrapper>& g_output_;
  vector<GradientWrapper> g_input_;
};

/**
 * @brief A helper class to indicate that the operator does not need gradient
 * computation.
 *
 * Use the macro NO_GRADIENT to register operators that do not have gradients.
 * Note that this is different fron SHOULD_NOT_DO_GRADIENT: the latter means
 * that the gradient computation should not flow through it at all, and throws
 * an error if it is called.
 */
class CAFFE2_API NoGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return vector<OperatorDef>();
  }
};

/**
 * @brief A helper class to indicate that the operator should have no gradient.
 *
 * This is used when the operator definition is designed to not have a gradient.
 * Calling a gradient on this operator def will cause Caffe2 to quit.
 */
struct ThrowInTheTowelIfGradientIsCalled : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  GradientOpsMeta Get() override {
    CAFFE_ENFORCE(
        false, "One should not call gradient for operator ", def_.type(), ".");
  }
};

/**
 * @brief A helper class to indicate that the gradient mechanism is not ready.
 *
 * This should only be used sparsely when the gradient does exist, but we have
 * not implemented it yet and are using this as a lazy excuse. Eventually, a
 * gradient operator should be implemented.
 */
struct GradientNotImplementedYet : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  GradientOpsMeta Get() override {
    CAFFE_ENFORCE(
        false,
        "Operator ",
        def_.type(),
        " should have a gradient but is not implemented yet.");
  }
};

C10_DECLARE_REGISTRY(
    GradientRegistry,
    GradientMakerBase,
    const OperatorDef&,
    const vector<GradientWrapper>&);

#ifdef CAFFE2_NO_GRADIENT_OPS

#define REGISTER_GRADIENT(name, ...) /* No gradients. */
#define REGISTER_GRADIENT_STR(str_name, ...) /* No gradients. */

#else

#define REGISTER_GRADIENT(name, ...) \
  C10_REGISTER_CLASS(GradientRegistry, name, __VA_ARGS__)
#define REGISTER_GRADIENT_STR(str_name, ...) \
  C10_REGISTER_TYPED_CLASS(GradientRegistry, str_name, __VA_ARGS__)

#endif

// NO_GRADIENT means that the operator does not need any gradient computation.
#define NO_GRADIENT(name) REGISTER_GRADIENT(name, NoGradient)

// SHOULD_NOT_DO_GRADIENT means that the operator is not designed to have
// gradient operators. If you attempt to call the gradient, a log fatal will
// occur.
#define SHOULD_NOT_DO_GRADIENT(name) \
  REGISTER_GRADIENT(name, ThrowInTheTowelIfGradientIsCalled)

#define GRADIENT_NOT_IMPLEMENTED_YET(name) \
  REGISTER_GRADIENT(name, GradientNotImplementedYet)

/**
 * @brief Gets the GradientOpsMeta for the given operator def.
 */
CAFFE2_API GradientOpsMeta GetGradientForOp(
    const OperatorDef& def,
    const vector<GradientWrapper>& g_output);

} // namespace caffe2

#endif // CAFFE2_CORE_OPERATOR_GRADIENT_H_
