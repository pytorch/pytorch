#include "caffe2/operators/pow_op.h"
#include "caffe2/utils/math.h"
// definition of NumericTypes and SameTypeAsInput is in below header file
//#include "caffe2/operators/elementwise_op.h"
#include <Eigen/Core>

namespace caffe2 {

#define EIGEN_POW(x, y) (x.pow(y))

struct EigenPowFunctor {
  template <int b_is_scalar, typename T1, typename T2, typename R>
  inline void
  Run(size_t n, const T1* a, const T2* b, T2 e, R* out, CPUContext*) {
    if (b == NULL) {
      EigenVectorArrayMap<R>(out, n) =
          EIGEN_POW((ConstEigenVectorArrayMap<T1>(a, n)), (e));
    } else {
      if (b_is_scalar) {
        EigenVectorArrayMap<R>(out, n) =
            EIGEN_POW((ConstEigenVectorArrayMap<T1>(a, n)), (b[0]));
      } else {
        EigenVectorArrayMap<R>(out, n) = EIGEN_POW(
            (ConstEigenVectorArrayMap<T1>(a, n)),
            (ConstEigenVectorArrayMap<T2>(b, n)));
      }
    }
  }
  template <typename T1, typename T2, typename R>
  void RunWithBroadcast(
      const T1* a,
      const T2* b,
      R* out,
      size_t pre,
      size_t n,
      CPUContext*) {
    EigenArrayMap<R>(out, n, pre) = EIGEN_POW(
        (ConstEigenArrayMap<T1>(a, n, pre)),
        (ConstEigenVectorArrayMap<T2>(b, n)).rowwise().replicate(pre));
    /*
    //below code only allows elementary ops, such as +, -, * and /,
    //and does not allow operations, such as pow, exp and log
    EIGEN_POW(
       (ConstEigenArrayMap<T>(a, n, pre).colwise()),
       (ConstEigenVectorArrayMap<T>(b, n)));
     */
  }
  template <typename T1, typename T2, typename R>
  void RunWithBroadcast2(
      const T1* a,
      const T2* b,
      R* out,
      size_t pre,
      size_t n,
      size_t post,
      CPUContext*) {
    for (int i = 0; i < pre; ++i) {
      EigenArrayMap<R>(out + i * n * post, post, n) = EIGEN_POW(
          (ConstEigenArrayMap<T1>(a + i * n * post, post, n)),
          (Eigen::Map<const Eigen::Array<T2, 1, Eigen::Dynamic>>(b, n))
              .colwise()
              .replicate(post));
      /*
      //below code only allows elementary ops, such as +, -, * and /,
      //and does not allow for operations, such as pow, exp and log
      EIEGN_POW(
        (ConstEigenArrayMap<T>(a + i * n * post, post, n).rowwise()),
        (Eigen::Map<const Eigen::Array<T, 1, Eigen::Dynamic>>(b, n)));
      */
    }
  }
};

REGISTER_CPU_OPERATOR(
    Pow,
    PowOp<
        TensorTypes<float> /*NumericTypes*/,
        CPUContext,
        EigenPowFunctor,
        SameTypeAsInput>)

OPERATOR_SCHEMA(Pow)
    .NumInputs(1, 2)
    .NumOutputs(1)
    .Arg("exponent", "The exponent of the power function.")
    .AllowInplace({{0, 0}, {1, 0}})
    .IdenticalTypeAndShapeOfInput(0)
    .SetDoc(R"DOC(
Pow takes input data (Tensor<T>) and an argument exponent, which can be a
scalar or another tensor. It produces one output data (Tensor<T>), where
the function `f(x) = x^exponent` is applied to the data tensor elementwise.
)DOC")
    .Input(0, "X", "Input tensor of any shape")
    .Input(1, "exponent", "The exponent of the power function.")
    .Output(0, "Y", "Output tensor (same size as X)");

class GetPowGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    ArgumentHelper arg_helper(def_);
    if (arg_helper.HasArgument("exponent")) { // second input is a scalar
      // function f(w,a) = w^a
      // gradient operator with respect to first input tensor
      // df/dw = a * w^(a-1) (all operations are component-wise)
      float exponent = arg_helper.GetSingleArgument<float>("exponent", 0.0);
      Argument scale_arg;
      scale_arg.set_name("scale");
      scale_arg.set_f(exponent);
      Argument pow_arg;
      pow_arg.set_name("exponent");
      if (I(0) != O(0)) {
        pow_arg.set_f(exponent - 1);
      } else {
        LOG(WARNING) << "In-place Pow gradient, possible loss of precision";
        constexpr float kEps = 1e-12f;
        CAFFE_ENFORCE(std::fabs(exponent) > kEps);
        pow_arg.set_f((exponent - 1) / exponent);
      }
      return vector<OperatorDef>{CreateOperatorDef(
                                     "Pow",
                                     "",
                                     std::vector<string>{I(0)},
                                     std::vector<string>{GI(0)},
                                     std::vector<Argument>{pow_arg}),
                                 CreateOperatorDef(
                                     "Mul",
                                     "",
                                     std::vector<string>{GI(0), GO(0)},
                                     std::vector<string>{GI(0)}),
                                 CreateOperatorDef(
                                     "Scale",
                                     "",
                                     std::vector<string>{GI(0)},
                                     std::vector<string>{GI(0)},
                                     std::vector<Argument>{scale_arg})};
      /*
      // Alternative gradient computation
      return vector<OperatorDef>{CreateOperatorDef(
                                     "Div",
                                     "",
                                     std::vector<string>{O(0), I(0)},
                                     std::vector<string>{GI(0)}),
                                 CreateOperatorDef(
                                     "Mul",
                                     "",
                                     std::vector<string>{GI(0), GO(0)},
                                     std::vector<string>{GI(0)}),
                                 CreateOperatorDef(
                                     "Scale",
                                     "",
                                     std::vector<string>{GI(0)},
                                     std::vector<string>{GI(0)},
                                     std::vector<Argument>{scale_arg})};
      */
    } else { // second input is a tensor
      CAFFE_ENFORCE(
          Def().input(0) != Def().output(0) &&
              Def().input(1) != Def().output(0),
          "Gradient computation cannot be carried out if Pow uses in-place "
          "computation: ",
          ProtoDebugString(Def()));
      vector<OperatorDef> grad_ops;
      Argument one_arg;
      one_arg.set_name("value");
      one_arg.set_f(1);
      Argument broadcast, axis, axis_str, order;
      bool bflag = ArgumentHelper::HasArgument(Def(), "broadcast");

      if (bflag) {
        if (ArgumentHelper::HasArgument(Def(), "broadcast")) {
          broadcast = GetArgument(Def(), "broadcast");
        } else {
          broadcast = MakeArgument<int>("broadcast", 0);
        }
        if (ArgumentHelper::HasArgument(Def(), "axis")) {
          axis = GetArgument(Def(), "axis");
        } else {
          axis = MakeArgument<int>("axis", -1);
        }
        if (ArgumentHelper::HasArgument(Def(), "axis_str")) {
          axis_str = GetArgument(Def(), "axis_str");
        } else {
          axis_str = MakeArgument<string>("axis_str", "");
        }
        if (ArgumentHelper::HasArgument(Def(), "order")) {
          order = GetArgument(Def(), "order");
        } else {
          order = MakeArgument<string>("order", "NCHW");
        }
      }

      // function f(w,a) = w^a
      // gradient operator with respect to first input tensor
      // df/dw = a * w^(a-1) (all operations are component-wise)
      grad_ops.push_back(CreateOperatorDef(
          "ConstantFill",
          "",
          std::vector<string>{I(1)},
          std::vector<string>{GI(1)},
          std::vector<Argument>{one_arg}));
      grad_ops.push_back(CreateOperatorDef(
          "Sub",
          "",
          std::vector<string>{I(1), GI(1)},
          std::vector<string>{GI(1)}));
      if (bflag) {
        grad_ops.push_back(CreateOperatorDef(
            "Pow",
            "",
            std::vector<string>{I(0), GI(1)},
            std::vector<string>{GI(0)},
            vector<Argument>{broadcast, axis, axis_str, order}));
      } else {
        grad_ops.push_back(CreateOperatorDef(
            "Pow",
            "",
            std::vector<string>{I(0), GI(1)},
            std::vector<string>{GI(0)}));
      }

      grad_ops.push_back(CreateOperatorDef(
          "Mul",
          "",
          std::vector<string>{GI(0), GO(0)},
          std::vector<string>{GI(0)}));
      if (bflag) {
        grad_ops.push_back(CreateOperatorDef(
            "Mul",
            "",
            std::vector<string>{GI(0), I(1)},
            std::vector<string>{GI(0)},
            vector<Argument>{broadcast, axis, axis_str, order}));
      } else {
        grad_ops.push_back(CreateOperatorDef(
            "Mul",
            "",
            std::vector<string>{GI(0), I(1)},
            std::vector<string>{GI(0)}));
      }
      /*
      // Alternative gradient computation (no broadcast support)
      grad_ops.push_back(CreateOperatorDef(
                           "Div",
                           "",
                           std::vector<string>{O(0), I(0)},
                           std::vector<string>{GI(0)}));
      grad_ops.push_back(CreateOperatorDef(
                           "Mul",
                           "",
                           std::vector<string>{GI(0), GO(0)},
                           std::vector<string>{GI(0)}));
      grad_ops.push_back(CreateOperatorDef(
                           "Mul",
                           "",
                           std::vector<string>{GI(0), I(1)},
                           std::vector<string>{GI(0)}));
      */
      // gradient operator for with respect to second input tensor
      // df/da =  w^a * ln w (all operations are component-wise)
      /*
      // reset GI(1) to zero
      Argument zero_arg;
      zero_arg.set_name("value");
      zero_arg.set_f(0);
      grad_ops.push_back(CreateOperatorDef(
          "ConstantFill",
          "",
          std::vector<string>{I(1)},
          std::vector<string>{GI(1)},
          std::vector<Argument>{zero_arg}));
      */
      grad_ops.push_back(CreateOperatorDef(
          "Log",
          "",
          std::vector<string>{I(0)},
          std::vector<string>{GI(1) + "_autogen_pre_red"}));
      grad_ops.push_back(CreateOperatorDef(
          "Mul",
          "",
          std::vector<string>{GI(1) + "_autogen_pre_red", O(0)},
          std::vector<string>{GI(1) + "_autogen_pre_red"}));
      if (bflag) {
        grad_ops.push_back(CreateOperatorDef(
            "Mul",
            "",
            std::vector<string>{GI(1) + "_autogen_pre_red", GO(0)},
            std::vector<string>{GI(1) + "_autogen_pre_red"}));
        grad_ops.push_back(CreateOperatorDef(
            "SumReduceLike",
            "",
            vector<string>{GI(1) + "_autogen_pre_red", I(1)},
            vector<string>{GI(1)},
            vector<Argument>{axis, axis_str, order}));
      } else {
        grad_ops.push_back(CreateOperatorDef(
            "Mul",
            "",
            std::vector<string>{GI(1) + "_autogen_pre_red", GO(0)},
            std::vector<string>{GI(1)}));
      }

      return grad_ops;
    }
  }

  // Argument `shape` is no longer needed in backprop.
  bool CopyArguments() const override {
    return false;
  }
};

REGISTER_GRADIENT(Pow, GetPowGradient);

} // namespace caffe2
