#include "THCUNN.h"
#include "THCAtomics.cuh"
#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCDeviceUtils.cuh"
#include "THCNumerics.cuh"
#include "THCTensorTypeUtils.cuh"

#define OUTPUT_FEATURES_PER_THREAD 32
#define MAX_WARPS_PER_RUN 4

namespace detail {

/// Various utilities for dealing with arrays of values which are
/// maintained in thread-local registers. All accesses are done in such
/// a way such that the index is statically known, which preserves the
/// compiler's ability to allocate the values to registers, as opposed
/// to local memory.
template <typename T, int N>
struct RegisterUtils {
  /// Register shifting: move elements towards the beginning of the
  /// array (towards 0) by `Shift` places:
  /// arr[i] = arr[i + Shift]
  /// The `Shift` elements at the end are left unchanged.
  template <int Shift>
  __device__ __forceinline__ static void shiftLeft(T arr[N]) {
    // e.g., N = 5, Shift = 2:
    // 0 1 2 3 4 becomes =>
    // 2 3 4 3 4 (last are unchanged)
#pragma unroll
    for (int i = 0; i < N - Shift; ++i) {
      arr[i] = arr[i + Shift];
    }
  }
};

template <typename T>
__device__ __forceinline__
int getDim1Point(const THCDeviceTensor<T, 4>& input) {
  int threadPoint = blockIdx.x * blockDim.x + threadIdx.x;
  return threadPoint / input.getSize(3);
}

template <typename T>
__device__ __forceinline__
int getDim2Point(const THCDeviceTensor<T, 4>& input) {
  int threadPoint = blockIdx.x * blockDim.x + threadIdx.x;
  return threadPoint % input.getSize(3);
}

__device__ __forceinline__
int getStartOutputFeature() {
  return blockIdx.y * OUTPUT_FEATURES_PER_THREAD;
}

template <typename T>
__device__ __forceinline__
int getEndOutputFeature(const THCDeviceTensor<T, 4>& output) {
  return min((blockIdx.y + 1) * OUTPUT_FEATURES_PER_THREAD, output.getSize(1));
}

__device__ __forceinline__
int getBatch() {
  return blockIdx.z;
}

// All of these functions that follow are MathOps; they are template
// parameters so L2 can be more efficiently implemented
// template <typename T>
// typedef T (*MathOp)(const T in, const T arg);

template <typename T>
__device__ __forceinline__ T power2(const T in, const T power) {
  return THCNumerics<T>::mul(in, in);
}

template <typename T>
__device__ __forceinline__ T root2(const T in, const T power) {
  return THCNumerics<T>::sqrt(in);
}

template <typename T>
__device__ __forceinline__ T powerGrad2(const T in, const T power) {
  return in;
}

template <typename T>
__device__ __forceinline__ T powerN(const T in, const T power) {
  return THCNumerics<T>::pow(in, power);
}

template <typename T>
__device__ __forceinline__ T rootN(const T in, const T power) {
  const T invPower = THCNumerics<T>::cinv(power);
  return THCNumerics<T>::pow(in, invPower);
}

template <typename T>
__device__ __forceinline__ T powerGradN(const T in, const T power) {
  return THCNumerics<T>::pow(in,
                             THCNumerics<T>::sub(power,
                                                 ScalarConvert<int, T>::to(1)));
}

// Input is of the form:
// [batch][feature dim][optional dim 1][optional dim 2]
template <typename T,
          int Width,
          int Stride,
          T (*PowerFunc)(T in, T power),
          T (*RootFunc)(T in, T power)>
__global__ void
featureLPPoolingUpdateOutput(const THCDeviceTensor<T, 4> input,
                             THCDeviceTensor<T, 4> output,
                             T power) {
  // What non-feature points is this thread handling?
  int dim1Point = getDim1Point(input);
  int dim2Point = getDim2Point(input);

  if (dim1Point >= input.getSize(2) || dim2Point >= input.getSize(3)) {
    // This thread in the warp is out of bounds
    return;
  }

  // What feature points is this thread handling?
  int startOutputFeature = getStartOutputFeature();
  int endOutputFeature = getEndOutputFeature(output);
  int startInputFeature = startOutputFeature * Stride;

  // What batch points is this thread handling?
  int batch = getBatch();

  // If stride >= width, then there is no loaded data reuse.
  // If stride > 1 and stride < width, then shift by stride, since we
  // can reuse Width - Stride elements from the previous round.
  // e.g., width = 5, stride = 2,
  // output 0 uses input 0 1 2 3 4
  // output 1 uses input 2 3 4 5 6 (inputs 2 - 4 are reused, i.e., 5 -
  // 2 elements are reused, and we have to shift the array by 2)
  //
  // e.g., width = 5, stride = 3,
  // output 0 uses input 0 1 2 3 4
  // output 1 uses input 3 4 5 6 7 (inputs 3 - 4 are reused, i.e., 5 - 3
  // elements are reused, and we have to shift the array by 3)

  // Valid only pooling: load Width elements from input (Width -
  // Stride is handled here, at the top of the loop we handle the
  // remaining Stride elements). We already verified that the input is
  // larger than the width.
  // `in` will contain the input values ^ power.
  T in[Width];

#pragma unroll
  for (int i = 0; i < Width - Stride; ++i) {
    const T data =
      input[batch][startInputFeature + i][dim1Point][dim2Point];
    in[i] = PowerFunc(data, power);
  }

  for (int outputFeature = startOutputFeature;
       outputFeature < endOutputFeature;
       ++outputFeature) {
    // If Stride < Width, we're loading Stride new values starting at
    // Width - Stride
    // If Stride >= Width, we're loading Width new values starting at 0
    if (Stride < Width) {
      int nextInputFeature = outputFeature * Stride + Width - Stride;

#pragma unroll
      for (int i = 0; i < Stride; ++i) {
        const T data =
          input[batch][nextInputFeature + i][dim1Point][dim2Point];
        in[Width - Stride + i] = PowerFunc(data, power);
      }
    } else {
      int nextInputFeature = outputFeature * Stride;

#pragma unroll
      for (int i = 0; i < Width; ++i) {
        T data = input[batch][nextInputFeature + i][dim1Point][dim2Point];
        in[i] = PowerFunc(data, power);
      }
    }

    // Calculate the new output feature
    T val = ScalarConvert<int, T>::to(0);
    for (int i = 0; i < Width; ++i) {
      val = THCNumerics<T>::add(val, in[i]);
    }

    val = RootFunc(val, power);
    output[batch][outputFeature][dim1Point][dim2Point] = val;

    if (Stride < Width) {
      // Shift registers for calculating the next point
      RegisterUtils<T, Width>::shiftLeft<Stride>(in);
    }
  }
}

// forward pass: f(a, ..., z) = (a^p + ... + z^p)^(1 / p)
// for bprop:
//   partial df(a, ... z)/da = a^(p - 1) * (a^p + ... + z^p)^((1 / p) - 1) =
//   a^(p - 1) * 1/(f(a, ..., z)^(p - 1)) = (a / f(a, ..., z))^(p - 1)
//
// example: for p = 2, df(a, ..., z)/da = a / f(a, ..., z)
// example: for p = 3, df(a, ..., z)/da = (a / f(a, ..., z))^2
//
// PowerGradFunc implements x^(p - 1)
template <typename T,
          int Width,
          int Stride,
          T (*PowerGradFunc)(T in, T arg)>
__global__ void
featureLPPoolingUpdateGradInput(const THCDeviceTensor<T, 4> gradOutput,
                                const THCDeviceTensor<T, 4> input,
                                const THCDeviceTensor<T, 4> output,
                                THCDeviceTensor<T, 4> gradInput,
                                T power) {
  // What non-feature points is this thread handling?
  int dim1Point = getDim1Point(input);
  int dim2Point = getDim2Point(input);

  if (dim1Point >= input.getSize(2) || dim2Point >= input.getSize(3)) {
    // This thread in the warp is out of bounds
    return;
  }

  // What feature points is this thread handling? [start, end)
  int startOutputFeature = getStartOutputFeature();
  int endOutputFeature = getEndOutputFeature(output);

  // What is the first input point that the output features depend
  // upon? [start, end)
  int startInputFeature = startOutputFeature * Stride;
  int endInputFeature = endOutputFeature * Stride;

  // What batch points is this thread handling?
  int batch = getBatch();

  // atomicAdd into gradInput is slow, avoid it where possible.
  // We can do this because there is a range of gradInput elements
  // that we are updating exclusively. This is how we find it
  //
  //  width = 3 stride = 1 example:
  // ------------------------------
  //      startOutputFeature for this thread
  //        |
  //        |
  // previous thread's output feature
  //   |    |
  //   |    |                  gradOutput
  // __v____v___________________
  // |    |    |    |    |    |
  // ---------------------------
  //   |\ \_____
  //   | \__    \               gradInput
  // __v____v____v_____________
  // |    |    |    |    |    |
  // ---------------------------
  //         A        A
  //         |        |
  //    startInputFeature
  //                  |
  //                  exclusiveStartInputFeature
  //
  // exclusiveStartInputFeature is the first input feature that we can
  // write into exclusively; the one right before it overlaps with
  // updates from a previous thread and thus has to use atomicAdd.
  int exclusiveStartInputFeature =
    startInputFeature == 0 ?
    // no thread is before ourselves
    0 :
    // there is a thread before ourselves
    startInputFeature + (Width - 1) * Stride;

  // Similarly, exclusiveEndInputFeature is the last input feature
  // that we can write into exclusively, since we might be overlapping
  // with the following thread
  int exclusiveEndInputFeature =
    endOutputFeature == output.getSize(1) ?
    // no thread is after ourselves
    endInputFeature + (Width - 1) * Stride :
    // there is a thread after ourselves
    endInputFeature;

  // As with updateOutput preload input elements, except no need to
  // transform them
  T in[Width];
#pragma unroll
  for (int i = 0; i < Width - Stride; ++i) {
    in[i] = input[batch][startInputFeature + i][dim1Point][dim2Point];
  }

  for (int outputFeature = startOutputFeature;
       outputFeature < endOutputFeature;
       ++outputFeature) {
    // As with updateOutput load the subsequent input elements that we
    // need, except no need to transform them
    //
    // If Stride < Width, we're loading Stride new values starting at
    // Width - Stride
    // If Stride >= Width, we're loading Width new values starting at 0
    if (Stride < Width) {
      int nextInputFeature = outputFeature * Stride + Width - Stride;

#pragma unroll
      for (int i = 0; i < Stride; ++i) {
        in[Width - Stride + i] =
          input[batch][nextInputFeature + i][dim1Point][dim2Point];
      }
    } else {
      int nextInputFeature = outputFeature * Stride;

#pragma unroll
      for (int i = 0; i < Width; ++i) {
        in[i] = input[batch][nextInputFeature + i][dim1Point][dim2Point];
      }
    }

    // A given output feature gradient contributes to `Width` input
    // gradients
    const T gradOut =
      gradOutput[batch][outputFeature][dim1Point][dim2Point];

    // Load output (f(x_is)). It is possible that this is zero, in
    // which case we'll ignore this point.
    T out = output[batch][outputFeature][dim1Point][dim2Point];
    if (THCNumerics<T>::eq(out, ScalarConvert<int, T>::to(0))) {
      continue;
    }

    int curStartInputFeature = outputFeature * Stride;
    int curEndInputFeature = outputFeature * Stride + Width - 1;

    if (curStartInputFeature >= exclusiveStartInputFeature &&
        curEndInputFeature < exclusiveEndInputFeature) {
      // This thread is exclusively responsible for updating these
      // input points, so we need not make the addition atomic
      for (int i = 0; i < Width; ++i) {
        int inputFeature = outputFeature * Stride + i;

        // Calculate grad * (x_i / f(x_is))^(p - 1)
        const T val = THCNumerics<T>::mul(
          gradOut,
          PowerGradFunc(THCNumerics<T>::div(in[i], out), power));

        gradInput[batch][inputFeature][dim1Point][dim2Point] =
          THCNumerics<T>::add(
            gradInput[batch][inputFeature][dim1Point][dim2Point], val);
      }
    } else {
      // Handle start and end boundary cases: potential overlap with
      // other threads
      for (int i = 0; i < Width; ++i) {
        int inputFeature = outputFeature * Stride + i;

        // Calculate grad * (x_i / f(x_is))^(p - 1)
        T val = THCNumerics<T>::mul(
          gradOut,
          PowerGradFunc(THCNumerics<T>::div(in[i], out), power));

        // We don't overlap other threads for this range
        if (inputFeature >= exclusiveStartInputFeature &&
            inputFeature < exclusiveEndInputFeature) {
          gradInput[batch][inputFeature][dim1Point][dim2Point]
            = THCNumerics<T>::add(
              gradInput[batch][inputFeature][dim1Point][dim2Point], val);
        } else {
          // We are potentially overlapping with threads handling
          // features before ourselves, so these need to be added atomically
          atomicAdd(&gradInput[batch][inputFeature][dim1Point][dim2Point],
                    val);
        }
      }
    }

    if (Stride < Width) {
      // Shift registers for calculating the next point
      RegisterUtils<T, Width>::shiftLeft<Stride>(in);
    }
  }
}

} // namespace detail

inline int lpPoolingOutputSize(int inputSize, int width, int stride) {
  return ((inputSize - width) / stride) + 1;
}

template <typename T>
bool
runFeatureLPPoolingUpdateOutput(THCState* state,
                                const THCDeviceTensor<T, 4>& input,
                                THCDeviceTensor<T, 4>& output,
                                float power, int width, int stride) {
  cudaStream_t stream =
    THCState_getCurrentStream(state);
  const cudaDeviceProp* deviceProperties =
    THCState_getCurrentDeviceProperties(state);

  int outputFeatures = ((input.getSize(1) - width) / stride) + 1;

  THAssert(input.getSize(0) == output.getSize(0));
  THAssert(outputFeatures == output.getSize(1));
  THAssert(input.getSize(1) >= width);

  THAssert(input.getSize(2) == output.getSize(2));
  THAssert(input.getSize(3) == output.getSize(3));
  THAssert(power > 0.0f);
  THAssert(width >= 1);
  THAssert(stride >= 1);

  // Split non-features among threads and grid x
  int totalNonFeatureSize = input.getSize(2) * input.getSize(3);
  int numWarps =
    min(THCCeilDiv(totalNonFeatureSize, deviceProperties->warpSize),
        MAX_WARPS_PER_RUN);
  int blockSize = deviceProperties->warpSize * numWarps;

  // Split non-features among grid x
  int nonFeatureSizeBlocks = THCCeilDiv(totalNonFeatureSize, blockSize);

  // Split features among grid y, up to a maximum number of features per thread
  int featureBlocks = THCCeilDiv(outputFeatures, OUTPUT_FEATURES_PER_THREAD);

  // Split batch among grid z.
  dim3 grid(nonFeatureSizeBlocks, featureBlocks, input.getSize(0));
  dim3 block(blockSize);

#define L2_STRIDE_CASE(STRIDE, WIDTH)                                   \
  case STRIDE:                                                          \
    detail::                                                            \
    featureLPPoolingUpdateOutput<T, WIDTH,                              \
                                 STRIDE,                                \
                                 detail::power2,                        \
                                 detail::root2><<<grid, block, 0, stream>>>( \
                                   input, output,                       \
                                   ScalarConvert<float, T>::to(power)); \
    return true;

#define L2_WIDTH_CASE(WIDTH)                    \
  case WIDTH:                                   \
    switch (stride) {                           \
      L2_STRIDE_CASE(1, WIDTH);                 \
      L2_STRIDE_CASE(2, WIDTH);                 \
      L2_STRIDE_CASE(3, WIDTH);                 \
      L2_STRIDE_CASE(4, WIDTH);                 \
    }

#define LP_STRIDE_CASE(STRIDE, WIDTH)                                   \
  case STRIDE:                                                          \
    detail::                                                            \
    featureLPPoolingUpdateOutput<T, WIDTH,                              \
                                 STRIDE,                                \
                                 detail::powerN,                        \
                                 detail::rootN><<<grid, block, 0, stream>>>( \
                                   input, output,                       \
                                   ScalarConvert<float, T>::to(power)); \
    return true;

#define LP_WIDTH_CASE(WIDTH)                    \
  case WIDTH:                                   \
    switch (stride) {                           \
      LP_STRIDE_CASE(1, WIDTH);                 \
      LP_STRIDE_CASE(2, WIDTH);                 \
      LP_STRIDE_CASE(3, WIDTH);                 \
      LP_STRIDE_CASE(4, WIDTH);                 \
    }

  if (power == 2.0f) {
    switch (width) {
      L2_WIDTH_CASE(2);
      L2_WIDTH_CASE(3);
      L2_WIDTH_CASE(4);
      L2_WIDTH_CASE(5);
      L2_WIDTH_CASE(6);
      L2_WIDTH_CASE(7);
      L2_WIDTH_CASE(8);
      L2_WIDTH_CASE(9);
      L2_WIDTH_CASE(10);
      L2_WIDTH_CASE(11);
      L2_WIDTH_CASE(12);
      L2_WIDTH_CASE(13);
      L2_WIDTH_CASE(14);
      L2_WIDTH_CASE(15);
      L2_WIDTH_CASE(16);
    }
  } else {
    switch (width) {
      LP_WIDTH_CASE(2);
      LP_WIDTH_CASE(3);
      LP_WIDTH_CASE(4);
      LP_WIDTH_CASE(5);
      LP_WIDTH_CASE(6);
      LP_WIDTH_CASE(7);
      LP_WIDTH_CASE(8);
      LP_WIDTH_CASE(9);
      LP_WIDTH_CASE(10);
      LP_WIDTH_CASE(11);
      LP_WIDTH_CASE(12);
      LP_WIDTH_CASE(13);
      LP_WIDTH_CASE(14);
      LP_WIDTH_CASE(15);
      LP_WIDTH_CASE(16);
    }
  }

  // Otherwise, we have an unhandled width and/or stride.
  return false;

#undef L2_STRIDE_CASE
#undef L2_WIDTH_CASE
#undef LP_STRIDE_CASE
#undef LP_WIDTH_CASE
}

template <typename T>
bool
runFeatureLPPoolingUpdateGradInput(THCState* state,
                                   const THCDeviceTensor<T, 4>& gradOutput,
                                   const THCDeviceTensor<T, 4>& input,
                                   const THCDeviceTensor<T, 4>& output,
                                   THCDeviceTensor<T, 4>& gradInput,
                                   float power, int width, int stride) {
  cudaStream_t stream =
    THCState_getCurrentStream(state);
  const cudaDeviceProp* deviceProperties =
    THCState_getCurrentDeviceProperties(state);

  for (int i = 0; i < 4; ++i) {
    THAssert(gradOutput.getSize(i) == output.getSize(i));
    THAssert(gradInput.getSize(i) == input.getSize(i));
  }

  int outputFeatures = ((input.getSize(1) - width) / stride) + 1;

  THAssert(gradInput.getSize(0) == gradOutput.getSize(0));
  THAssert(outputFeatures == gradOutput.getSize(1));
  THAssert(gradInput.getSize(1) >= width);

  THAssert(gradInput.getSize(2) == gradOutput.getSize(2));
  THAssert(gradInput.getSize(3) == gradOutput.getSize(3));
  THAssert(power > 0.0f);
  THAssert(width >= 1);
  THAssert(stride >= 1);

  // Different threads are potentially adding into overlapping input
  // points, so we must clear out gradInput before continuing.
  gradInput.zero(stream);

  // Split non-features among threads and grid x
  int totalNonFeatureSize = input.getSize(2) * input.getSize(3);
  int numWarps =
    min(THCCeilDiv(totalNonFeatureSize, deviceProperties->warpSize),
        MAX_WARPS_PER_RUN);
  int blockSize = deviceProperties->warpSize * numWarps;

  // Split non-features among grid x
  int nonFeatureSizeBlocks = THCCeilDiv(totalNonFeatureSize, blockSize);

  // Split features among grid y, up to a maximum number of features per thread
  int featureBlocks = THCCeilDiv(outputFeatures, OUTPUT_FEATURES_PER_THREAD);

  // Split batch among grid z.
  dim3 grid(nonFeatureSizeBlocks, featureBlocks, input.getSize(0));
  dim3 block(blockSize);

#define L2_STRIDE_CASE(STRIDE, WIDTH)                                   \
  case STRIDE:                                                          \
    detail::                                                            \
    featureLPPoolingUpdateGradInput<                                    \
          T, WIDTH, STRIDE, detail::powerGrad2><<<grid, block, 0, stream>>>( \
            gradOutput, input, output, gradInput,                       \
            ScalarConvert<float, T>::to(power));                        \
    return true;

#define L2_WIDTH_CASE(WIDTH)                    \
  case WIDTH:                                   \
    switch (stride) {                           \
      L2_STRIDE_CASE(1, WIDTH);                 \
      L2_STRIDE_CASE(2, WIDTH);                 \
      L2_STRIDE_CASE(3, WIDTH);                 \
      L2_STRIDE_CASE(4, WIDTH);                 \
    }

#define LP_STRIDE_CASE(STRIDE, WIDTH)                                   \
  case STRIDE:                                                          \
    detail::                                                            \
    featureLPPoolingUpdateGradInput<                                    \
          T, WIDTH, STRIDE, detail::powerGradN><<<grid, block, 0, stream>>>( \
            gradOutput, input, output, gradInput,                       \
            ScalarConvert<float, T>::to(power));                        \
    return true;

#define LP_WIDTH_CASE(WIDTH)                    \
  case WIDTH:                                   \
    switch (stride) {                           \
      LP_STRIDE_CASE(1, WIDTH);                 \
      LP_STRIDE_CASE(2, WIDTH);                 \
      LP_STRIDE_CASE(3, WIDTH);                 \
      LP_STRIDE_CASE(4, WIDTH);                 \
    }

  if (power == 2.0f) {
    switch (width) {
      L2_WIDTH_CASE(2);
      L2_WIDTH_CASE(3);
      L2_WIDTH_CASE(4);
      L2_WIDTH_CASE(5);
      L2_WIDTH_CASE(6);
      L2_WIDTH_CASE(7);
      L2_WIDTH_CASE(8);
      L2_WIDTH_CASE(9);
      L2_WIDTH_CASE(10);
      L2_WIDTH_CASE(11);
      L2_WIDTH_CASE(12);
      L2_WIDTH_CASE(13);
      L2_WIDTH_CASE(14);
      L2_WIDTH_CASE(15);
      L2_WIDTH_CASE(16);
    }
  } else {
    switch (width) {
      LP_WIDTH_CASE(2);
      LP_WIDTH_CASE(3);
      LP_WIDTH_CASE(4);
      LP_WIDTH_CASE(5);
      LP_WIDTH_CASE(6);
      LP_WIDTH_CASE(7);
      LP_WIDTH_CASE(8);
      LP_WIDTH_CASE(9);
      LP_WIDTH_CASE(10);
      LP_WIDTH_CASE(11);
      LP_WIDTH_CASE(12);
      LP_WIDTH_CASE(13);
      LP_WIDTH_CASE(14);
      LP_WIDTH_CASE(15);
      LP_WIDTH_CASE(16);
    }
  }

  // Otherwise, we have an unhandled width and/or stride.
  return false;

#undef L2_STRIDE_CASE
#undef L2_WIDTH_CASE
#undef LP_STRIDE_CASE
#undef LP_WIDTH_CASE
}

#include "generic/FeatureLPPooling.cu"
#include "THCGenerateFloatTypes.h"
