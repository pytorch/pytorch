#include "rmsprop_op.h"

#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <>
void rmsprop_update<CPUContext>(
    int N,
    const float* g,
    const float* ms,
    const float* mom,
    float* ng,
    float* nms,
    float* nmom,
    float decay,
    float momentum,
    float epsilon,
    const float* lr,
    CPUContext* /*context*/) {
  ConstEigenVectorArrayMap<float> gVec(g, N);
  ConstEigenVectorArrayMap<float> msVec(ms, N);
  ConstEigenVectorArrayMap<float> momVec(mom, N);
  // Update new mean square estimate
  EigenVectorArrayMap<float> nmsVec(nms, N);
  nmsVec = msVec + (1.0f - decay) * (gVec * gVec - msVec);
  // Update momentum estimate
  EigenVectorArrayMap<float> nmomVec(nmom, N);
  nmomVec = momVec * momentum + lr[0] * gVec / (epsilon + nmsVec).sqrt();
  // New gradient is the momentum
  EigenVectorArrayMap<float>(ng, N) = nmomVec;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(RmsProp, RmsPropOp<float, CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(RmsProp)
    .NumInputs(4)
    .NumOutputs(3)
    .AllowInplace({{0, 0}, {1, 1}, {2, 2}})
    .SetDoc(R"DOC(
Computes the RMSProp update
(http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf).
Concretely, given inputs (grad, mean_squares, mom, lr), computes:

    mean_squares_o = mean_squares + (1 - decay) * (square(grad) - mean_squares)
    mom_o = momentum * mom + lr * grad / sqrt(epsilon + mean_squares_o)
    grad_o = mom_o

Returns (grad_o, mean_squares_o, mom_o).
)DOC");
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(RmsProp);

}
